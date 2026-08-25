"""Custom rl_games network: fuse a CNN image branch with a proprioceptive state vector.

rl_games' stock builders are image-only (``A2CBuilder`` runs the CNN over the whole obs) or
image+reward+last_action (``A2CResnetBuilder``); neither fuses a CNN with a separate proprio
vector. This builder does, so the wrist RGB-D camera can be trained end-to-end with the policy.

The CNN's flattened features are passed through a small projection FC (``cnn.fc_size``, default
128) with a ReLU before being concatenated with the proprio vector. Without it the raw 3136 CNN
dims swamp the ~24 proprio dims (proprio becomes <1% of the fused input and is effectively ignored);
the projection brings the two branches to comparable width so the downstream LSTM/MLP can actually
use proprio.

It consumes the Dict observations produced by the Isaac Lab rl_games wrapper when
``concate_obs_groups=False`` (see ``rl_games_camera_ppo_cfg.yaml``): the policy network sees
``{"policy": <proprio>, "image": <H,W,C>}`` and the asymmetric central-value network sees
``{"critic": <state>}``. The image (the rank-3 entry) is run through a small Nature-style CNN; its
flattened features are concatenated with the proprio vector(s) and the result is handed to
rl_games' standard A2C network (LSTM + MLP + heads). With no image present (the critic), it is a
plain pass-through of the state vector, so one class serves both networks.

Implementation trick: we strip ``cnn`` from the params before calling the parent
``A2CBuilder.Network.__init__`` and give it a flat ``input_shape = (cnn_out + proprio_dim,)``. The
parent then builds the LSTM/MLP/heads for a flat vector and its ``actor_cnn`` is an empty
``Sequential`` (a pass-through). ``forward`` produces the fused vector and delegates to the parent,
reusing all of rl_games' RNN/MLP plumbing.
"""

import copy

import torch
import torch.nn as nn

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import A2CBuilder

# Nature-CNN default (DQN/Atari), sized for ~64x64 input. Overridable via the yaml ``cnn.convs``.
_DEFAULT_CONVS = [
    {"filters": 32, "kernel_size": 8, "strides": 4, "padding": 0},
    {"filters": 64, "kernel_size": 4, "strides": 2, "padding": 0},
    {"filters": 64, "kernel_size": 3, "strides": 1, "padding": 0},
]


class InsertionHybridBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        return InsertionHybridBuilder.Network(self.params, **kwargs)

    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            input_shape = kwargs["input_shape"]
            if not isinstance(input_shape, dict):
                raise TypeError(
                    f"InsertionHybridBuilder expects Dict observations (concate_obs_groups=False); "
                    f"got input_shape={input_shape!r}"
                )

            # Auxiliary-head config (optional). The labels for the aux heads ride in their own obs group
            # (`label_key`); it is privileged (the actor-unobservable grasp tilt + true hole gap), so it
            # is EXCLUDED from the policy input below and consumed only by the supervised aux loss.
            aux_cfg = params.get("aux_head")
            self._aux_label_key = (aux_cfg or {}).get("label_key")

            # The image is the rank-3 (H, W, C) group; everything else is a flat proprio vector (minus
            # the privileged aux-label group, which is never fed to the policy).
            # (Plain attributes only here -- nn.Module is not initialized until super().__init__.)
            self._image_key = None
            image_chw = None
            proprio_dim = 0
            for key, shape in input_shape.items():
                if key == self._aux_label_key:
                    continue  # privileged aux label: excluded from the policy input
                if len(shape) == 3:
                    self._image_key = key
                    image_chw = (int(shape[2]), int(shape[0]), int(shape[1]))  # HWC -> CHW
                else:
                    proprio_dim += int(shape[0])

            cnn_params = params.get("cnn")
            convs = (cnn_params or {}).get("convs") or _DEFAULT_CONVS
            # Optional pretrained backbone (end-to-end visuomotor): ResNet-18 and EfficientNet-B0 are
            # fully-frozen ImageNet encoders on plain 3-channel RGB. Only the downstream projection FC +
            # LSTM/MLP/aux heads train.
            backbone = (cnn_params or {}).get("backbone")
            self._image_backbone = backbone
            self._image_is_resnet = bool(self._image_key is not None and backbone == "resnet18")
            self._image_is_efficientnet = bool(self._image_key is not None and backbone == "efficientnet_b0")
            self._image_is_imagenet_encoder = self._image_is_resnet or self._image_is_efficientnet

            # Measure the CNN's flattened output with a throwaway module (cannot store an nn.Module
            # before nn.Module.__init__ runs inside the parent constructor). ResNet-18's pooled feature
            # is always 512, so we skip the measurement (and a redundant weight download) for it.
            cnn_out = 0
            if self._image_key is not None:
                c, h, w = image_chw
                if self._image_is_resnet:
                    cnn_out = 512
                elif self._image_is_efficientnet:
                    cnn_out = 1280
                else:
                    with torch.no_grad():
                        cnn_out = self._make_cnn(c, convs)(torch.zeros(1, c, h, w)).flatten(1).shape[1]

            # Project the (large) CNN feature vector down before fusing with proprio, so proprio is
            # not drowned. With no image (the critic) there is no projection. Plain ints only here.
            self._cnn_out = cnn_out
            self._proj_dim = int((cnn_params or {}).get("fc_size", 128)) if self._image_key is not None else 0

            # EXPLICIT ESTIMATOR: aux targets listed in aux_head.feed_to_policy have their (detached)
            # PREDICTION concatenated into the policy input -- gives the policy a vision-predicted hole
            # pose instead of relying on the noisy 2.5cm socket anchor. Sized here so the parent MLP fits.
            self._feed_targets = list((aux_cfg or {}).get("feed_to_policy") or []) if self._image_key is not None else []
            _tg = (aux_cfg or {}).get("targets") or {}
            self._estimator_dim = sum(
                int(_tg[n]["slice"][1]) - int(_tg[n]["slice"][0]) for n in self._feed_targets if n in _tg
            )

            # Build the rest of the net as a flat (proj + proprio + estimator)-vector net: strip 'cnn' so
            # the parent does not build its own CNN, and pass the fused feature dim as the input shape.
            parent_params = copy.deepcopy(params)
            parent_params.pop("cnn", None)
            parent_params.pop("aux_head", None)
            parent_kwargs = dict(kwargs)
            parent_kwargs["input_shape"] = (self._proj_dim + proprio_dim + self._estimator_dim,)
            super().__init__(parent_params, **parent_kwargs)

            # Now that nn.Module is initialized, build the real CNN + projection FC that forward() uses.
            self._resnet_finetune_tail = False
            if self._image_key is not None:
                if self._image_is_resnet:
                    pretrained = bool((cnn_params or {}).get("pretrained", True))
                    weights_path = (cnn_params or {}).get("weights_path")
                    # Optional top-block (layer4) fine-tuning at a reduced effective LR (supervisor 2026-08-22).
                    self._resnet_finetune_tail = bool((cnn_params or {}).get("finetune_tail", False))
                    _ft_scale = float((cnn_params or {}).get("finetune_grad_scale", 0.1))
                    self._image_cnn = self._build_resnet18(
                        image_chw[0], pretrained, weights_path,
                        finetune_tail=self._resnet_finetune_tail, finetune_grad_scale=_ft_scale,
                    )
                elif self._image_is_efficientnet:
                    pretrained = bool((cnn_params or {}).get("pretrained", True))
                    weights_path = (cnn_params or {}).get("weights_path")
                    self._image_cnn = self._build_efficientnet_b0(image_chw[0], pretrained, weights_path)
                else:
                    self._image_cnn = self._make_cnn(image_chw[0], convs)
                self._image_proj = nn.Sequential(nn.Linear(self._cnn_out, self._proj_dim), nn.ReLU())
            else:
                self._image_cnn = None
                self._image_proj = None

            # Auxiliary supervised heads hang off the 128-d projected CNN features, so the aux gradient
            # flows ONLY into the image encoder (CNN + projection) -- forcing it to represent each cue --
            # and never into the proprio/LSTM/actor path. Each target is a small MLP regressor; its MSE
            # against the matching slice of the privileged label is returned by get_aux_loss() and added
            # to the PPO loss by the agent (rl_games' built-in model.get_aux_loss() hook).
            self._aux_loss = None
            self._aux_heads = None
            self._aux_spec = {}  # name -> (lo, hi, coef)
            if self._image_key is not None and aux_cfg:
                hidden = int(aux_cfg.get("hidden", 128))
                # Optional deeper regressor: aux_head.layers = number of hidden layers (default 1 -> the
                # original single-hidden-layer head, so existing checkpoints load unchanged). A deeper head
                # gives the explicit hole estimator more capacity to recover the hole from RGB when depth is
                # corrupted (the realistic-depth regime). Set aux_head.layers: 2 (+ a wider hidden) to use it.
                n_layers = max(1, int(aux_cfg.get("layers", 1)))
                heads = {}
                for name, spec in (aux_cfg.get("targets") or {}).items():
                    lo, hi = int(spec["slice"][0]), int(spec["slice"][1])
                    mods = [nn.Linear(self._proj_dim, hidden), nn.ReLU()]
                    for _ in range(n_layers - 1):
                        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
                    mods.append(nn.Linear(hidden, hi - lo))
                    heads[name] = nn.Sequential(*mods)
                    self._aux_spec[name] = (lo, hi, float(spec.get("coef", 1.0)))
                self._aux_heads = nn.ModuleDict(heads)

        @staticmethod
        def _make_cnn(in_channels, convs):
            layers = []
            ch = in_channels
            for cv in convs:
                layers.append(
                    nn.Conv2d(ch, cv["filters"], cv["kernel_size"], cv["strides"], cv.get("padding", 0))
                )
                layers.append(nn.ReLU())
                ch = cv["filters"]
            return nn.Sequential(*layers)

        @staticmethod
        def _build_resnet18(in_channels, pretrained, weights_path=None, finetune_tail=False, finetune_grad_scale=0.1):
            """ImageNet ResNet-18 encoder for plain 3-channel RGB, backbone frozen (optional layer4 fine-tune).

            - RGB-only: the wrist camera emits a normal 3-channel RGB frame (no depth, no frame-stack), which
              is exactly ImageNet's native input, so conv1 is used UNMODIFIED -- no channel inflation.
            - Every backbone param is frozen (conv1 included). fc is replaced by Identity so forward()
              returns the 512-d global-average-pooled feature; only the downstream projection FC + LSTM/MLP
              (and aux heads) train.
            - BatchNorm stays in eval mode at forward time (see forward()) so the frozen running stats are
              used, not noisy RL-minibatch stats. Robust to being offline: if the pretrained download fails
              we fall back to random init with a warning; an optional local ``weights_path`` (a torchvision
              resnet18 state_dict) is loaded first if given.
            """
            import warnings

            from torchvision.models import resnet18

            try:
                from torchvision.models import ResNet18_Weights

                default_weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception:  # very old torchvision
                default_weights = None

            net = None
            if pretrained:
                try:
                    net = resnet18(weights=default_weights)
                except Exception as exc:  # noqa: BLE001 (offline / download failure)
                    warnings.warn(f"[insertion_hybrid] ResNet-18 pretrained load failed ({exc}); random init.")
            if net is None:
                net = resnet18(weights=None)
                if weights_path:
                    try:
                        net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
                    except Exception as exc:  # noqa: BLE001
                        warnings.warn(f"[insertion_hybrid] local ResNet-18 weights load failed ({exc}); random init.")

            if in_channels != 3:
                raise ValueError(
                    f"insertion_hybrid ResNet-18 encoder is RGB-only (3ch); got in_channels={in_channels}. "
                    "Depth and frame-stack channel inflation were removed -- keep image_channels=3 and "
                    "frame_stack=1 (the LSTM carries temporal state)."
                )
            net.fc = nn.Identity()  # forward() returns the 512-d pooled feature
            # The ImageNet encoder is a fixed feature extractor. Frozen Parameters are intentionally kept as
            # Parameters so checkpoints retain the standard torchvision state layout; Adam ignores entries
            # whose gradients are None.
            for p in net.parameters():
                p.requires_grad_(False)
            if finetune_tail:
                # Unfreeze ONLY the top residual block (layer4) for task fine-tuning. rl_games uses a single
                # optimizer LR for all params, so we approximate a smaller backbone LR with a gradient-scale
                # hook (grad *= finetune_grad_scale). BN stays frozen in eval() at forward time (see forward),
                # so running stats never drift on RL minibatches -- this + torch_compile OFF avoids the
                # frozen-backbone in-place NaN history. forward() runs the frozen stem under no_grad and only
                # layer4+pool with grad, so activation memory only grows for the top block.
                for p in net.layer4.parameters():
                    p.requires_grad_(True)
                    if finetune_grad_scale != 1.0:
                        p.register_hook(lambda g, s=finetune_grad_scale: g * s)
            return net

        @staticmethod
        def _build_efficientnet_b0(in_channels, pretrained, weights_path=None):
            """Fully-frozen ImageNet EfficientNet-B0 encoder returning its 1280-d pooled feature."""
            import warnings

            from torchvision.models import efficientnet_b0

            try:
                from torchvision.models import EfficientNet_B0_Weights

                default_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            except Exception:  # very old torchvision
                default_weights = None

            net = None
            if pretrained:
                try:
                    net = efficientnet_b0(weights=default_weights)
                except Exception as exc:  # noqa: BLE001 (offline / download failure)
                    warnings.warn(f"[insertion_hybrid] EfficientNet-B0 pretrained load failed ({exc}); random init.")
            if net is None:
                net = efficientnet_b0(weights=None)
                if weights_path:
                    try:
                        net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
                    except Exception as exc:  # noqa: BLE001
                        warnings.warn(
                            f"[insertion_hybrid] local EfficientNet-B0 weights load failed ({exc}); random init."
                        )

            if in_channels != 3:
                raise ValueError(f"EfficientNet-B0 encoder is RGB-only (3ch); got in_channels={in_channels}.")
            net.classifier = nn.Identity()
            for p in net.parameters():
                p.requires_grad_(False)
            return net

        def _resnet_forward_finetune(self, img):
            """ResNet-18 forward with the frozen stem (conv1..layer3) under no_grad and the trainable
            layer4 + avgpool under grad. Only the layer3 activation is retained for the layer4 backward,
            so activation memory grows only for the top block. BN is already in eval() (see forward)."""
            net = self._image_cnn
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    x = net.conv1(img); x = net.bn1(x); x = net.relu(x); x = net.maxpool(x)
                    x = net.layer1(x); x = net.layer2(x); x = net.layer3(x)
                x = net.layer4(x)
                x = net.avgpool(x)
            return torch.flatten(x, 1)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            # Concatenate proprio groups (everything that is neither the image nor the privileged aux
            # label), preserving dict order.
            proprio = [v for k, v in obs.items() if k != self._image_key and k != self._aux_label_key]
            feat = torch.cat(proprio, dim=1) if proprio else None
            cnn_feat = None
            self._aux_loss = None
            self._estimator_pred = None  # last explicit-estimator prediction (for viz/logging)
            if self._image_cnn is not None:
                image_obs = obs[self._image_key]
                img = image_obs.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
                import os as _os_dbg
                _debug_nan = bool(_os_dbg.environ.get("DEBUG_NAN"))
                _input_raw_max = float(img.abs().max()) if _debug_nan else None
                if self._image_is_imagenet_encoder:
                    # The camera path emits RGB in [0, 1], then ImageNet-normalizes it, so valid values are
                    # bounded by roughly [-2.12, 2.64]. A few sparse rollout-buffer pixels have been observed
                    # with corrupted finite exponents (up to 1e22); contain them at this explicit interface.
                    img = torch.nan_to_num(img, nan=0.0, posinf=2.65, neginf=-2.12).clamp(-2.12, 2.65)
                _probe_weight = None
                _probe_name = None
                _first_weight = None
                if self._image_is_resnet:
                    _probe_weight = self._image_cnn.layer4[0].conv2.weight
                    _probe_name = "l4conv2"
                    _first_weight = self._image_cnn.conv1.weight
                elif self._image_is_efficientnet:
                    _probe_weight = self._image_cnn.features[8][0].weight
                    _probe_name = "features8conv"
                    _first_weight = self._image_cnn.features[0][0].weight
                _probe_pre = float(_probe_weight.abs().max()) if _debug_nan and _probe_weight is not None else None
                if self._image_is_imagenet_encoder:
                    # Keep the frozen backbone's BatchNorm on its ImageNet running stats regardless of the
                    # agent toggling model.train()/eval().
                    self._image_cnn.eval()
                if self._image_is_imagenet_encoder:
                    # Encode in BF16 (not fp16); bf16 keeps fp32's exponent range at fp16's memory cost.
                    if self._image_is_resnet and self._resnet_finetune_tail:
                        cnn_out = self._resnet_forward_finetune(img)  # frozen stem no_grad, layer4 with grad
                    else:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
                            cnn_out = self._image_cnn(img).flatten(1)
                else:
                    cnn_out = self._image_cnn(img).flatten(1)
                # The projection remains outside no_grad and trains on the detached encoder features.
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self._image_is_imagenet_encoder):
                    cnn_feat = self._image_proj(cnn_out)
                if _debug_nan:
                    self._dbg_cnn_out = cnn_out  # Raw backbone output for the NaN probe.
                    _n = getattr(self, "_dbg_fwd_n", 0) + 1
                    self._dbg_fwd_n = _n
                    _rawmax = float(cnn_out.abs().max())
                    _cfmax = float(cnn_feat.abs().max())
                    # One-shot: when the RAW backbone output explodes, immediately re-run the SAME img through
                    # the SAME module in a clean eager fp32 no-autocast no_grad context. If the re-run is ~6.5
                    # while the in-place forward was 1e31, the transient lives in the autocast/compile path.
                    if _rawmax > 1e4 and not getattr(self, "_dbg_reforward_done", False):
                        self._dbg_reforward_done = True
                        with torch.no_grad():
                            with torch.autocast("cuda", enabled=False):
                                _re = self._image_cnn(img.float()).flatten(1)
                        print(f"[cnn-dbg] RE-FORWARD (eager fp32, same img+module): "
                              f"in_place_raw={_rawmax:.3g} -> reforward={float(_re.abs().max()):.3g}", flush=True)
                    if _n <= 10 or _rawmax > 1e4:
                        _c1 = float(_first_weight.abs().max()) if _first_weight is not None else float("nan")
                        _pj = float(self._image_proj[0].weight.abs().max())
                        _probe_post = float(_probe_weight.abs().max()) if _probe_weight is not None else float("nan")
                        import torch.nn as _nn_dbg
                        _bnv = _bnm = 0.0
                        _bntrain = 0
                        for _m in self._image_cnn.modules():
                            if isinstance(_m, _nn_dbg.BatchNorm2d):
                                if _m.running_var is not None:
                                    _bnv = max(_bnv, float(_m.running_var.max()))
                                    _bnm = max(_bnm, float(_m.running_mean.abs().max()))
                                if _m.training:
                                    _bntrain += 1
                        print(f"[cnn-dbg] fwd={_n} input_raw={_input_raw_max:.3g} "
                              f"img_absmax={float(img.abs().max()):.3g} "
                              f"encoder_out={_rawmax:.3g} cnn_feat={_cfmax:.3g} "
                              f"{_probe_name}_w={_probe_pre:.3g}->{_probe_post:.3g} "
                              f"conv1_w={_c1:.3g} proj_w={_pj:.3g} | cnn.training={self._image_cnn.training} "
                              f"bn_in_train={_bntrain} bn_var_max={_bnv:.3g} bn_mean_absmax={_bnm:.3g}", flush=True)
                    # Dump the EXACT triggering frame (BCHW, as fed to the backbone) + the raw pre-conv
                    # obs image on the first explosion, so it can be replayed against a bare frozen
                    # resnet18 in isolation (scripts/repro_resnet_nan.py). One dump per process.
                    if _rawmax > 1e4 and not getattr(self, "_dbg_frame_dumped", False):
                        self._dbg_frame_dumped = True
                        _row_max = cnn_out.detach().float().abs().amax(dim=1)  # (B,) per-env backbone max
                        _bad = torch.nonzero(_row_max > 1e4, as_tuple=False).flatten()
                        _dump = {
                            "img_bchw": img.detach().float().cpu(),          # exact backbone input
                            "obs_image_bhwc": obs[self._image_key].detach().float().cpu(),  # pre-permute obs
                            "encoder_out": cnn_out.detach().float().cpu(),
                            "resnet_out": cnn_out.detach().float().cpu(),  # compatibility with replay script
                            "bad_rows": _bad.detach().cpu(),
                            "fwd_n": _n,
                            # Snapshot the backbone state so a failing run can be diffed against ImageNet.
                            "backbone_state": {k: v.detach().float().cpu()
                                               for k, v in self._image_cnn.state_dict().items()},
                            "first_conv_weight": _first_weight.detach().float().cpu(),
                        }
                        _p = f"/tmp/resnet_nan_frame_fwd{_n}.pt"
                        torch.save(_dump, _p)
                        print(f"[cnn-dbg] DUMPED triggering frame -> {_p} "
                              f"(bad_rows={_bad.tolist()} of B={img.shape[0]}, img shape={tuple(img.shape)})",
                              flush=True)
                        raise RuntimeError("Frozen image encoder produced an invalid-magnitude feature; dump saved.")
                if feat is not None:
                    cnn_feat = cnn_feat.to(feat.dtype)  # match proprio dtype for the cat below
                # Compute the aux loss only on training forwards (skipped during rollout/eval, where the
                # heads are unused) and only when the privileged label is present.
                if self.training and self._aux_heads is not None and self._aux_label_key in obs:
                    self._aux_loss = self._compute_aux_loss(cnn_feat, obs[self._aux_label_key])
                parts = [cnn_feat] if feat is None else [cnn_feat, feat]
                # EXPLICIT ESTIMATOR: append the DETACHED predicted hole pose(s) so the policy consumes a
                # vision-predicted socket location (trained by the supervised aux loss; detach keeps the
                # estimator gradient purely supervised, not shaped by the policy loss).
                if self._feed_targets and self._aux_heads is not None:
                    preds = [self._aux_heads[_n](cnn_feat).detach() for _n in self._feed_targets]
                    self._estimator_pred = torch.cat(preds, dim=1)  # exposed for viz (pred vs true hole gap)
                    parts = parts + preds
                feat = torch.cat(parts, dim=1)
            fused = dict(obs_dict)
            fused["obs"] = feat
            out = super().forward(fused)
            # One-shot NaN probe (DEBUG_NAN=1): identify whether the actor mean or central value head failed,
            # then print the normalized inputs and image stages that fed it.
            import os as _os
            if _os.environ.get("DEBUG_NAN") and not getattr(self, "_output_nan_probed", False):
                head_out = out[0]
                if not torch.isfinite(head_out).all():
                    self._output_nan_probed = True
                    head_name = "actor mu" if self._image_cnn is not None else "central value"

                    def _st(t):
                        if t is None:
                            return "None"
                        fin = torch.isfinite(t)
                        amax = float(t[fin].abs().max()) if bool(fin.any()) else float("nan")
                        return f"nonfinite={int((~fin).sum())}/{t.numel()} absmax_finite={amax:.3g}"

                    print(f"[nan-probe] {head_name.upper()} NON-FINITE at first occurrence:", flush=True)
                    for _k, _v in obs.items():
                        print(f"  norm_obs[{_k}]: {_st(_v)}", flush=True)
                    print(f"  resnet_out(pre-proj): {_st(getattr(self, '_dbg_cnn_out', None))}", flush=True)
                    print(f"  cnn_feat: {_st(cnn_feat)}", flush=True)
                    print(f"  fused_feat: {_st(feat)}", flush=True)
                    print(f"  {head_name}: {_st(head_out)}", flush=True)
            return out

        def _compute_aux_loss(self, cnn_feat, label):
            # float32 for the regression (cnn_feat may be bf16 under autocast). Per-target MEAN MSE so a
            # 3-d head (hole) and a 1-d head (grasp) contribute on the same scale before their coef.
            feat = cnn_feat.float()
            label = label.float()
            return {
                name: coef * torch.nn.functional.mse_loss(self._aux_heads[name](feat), label[:, lo:hi])
                for name, (lo, hi, coef) in self._aux_spec.items()
            }

        def get_aux_loss(self):
            # rl_games' A2CAgent.calc_gradients adds each value of this dict to the PPO loss and logs it
            # under losses/<key>. None => no aux loss (e.g. the central-value network, or rollout/eval).
            return self._aux_loss


model_builder.register_network("insertion_hybrid", InsertionHybridBuilder)
