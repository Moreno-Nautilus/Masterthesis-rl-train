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
            # Optional pretrained backbone (end-to-end visuomotor): cnn.backbone == "resnet18" swaps the
            # from-scratch Nature CNN for a frozen ImageNet ResNet-18 whose input conv is inflated to the
            # image's channel count (RGB weights copied; extra channels = mean of the RGB weights) and
            # kept TRAINABLE so depth adapts. Everything else (LSTM/MLP/heads, projection FC) is unchanged.
            backbone = (cnn_params or {}).get("backbone")
            self._image_is_resnet = bool(self._image_key is not None and backbone == "resnet18")

            # Measure the CNN's flattened output with a throwaway module (cannot store an nn.Module
            # before nn.Module.__init__ runs inside the parent constructor). ResNet-18's pooled feature
            # is always 512, so we skip the measurement (and a redundant weight download) for it.
            cnn_out = 0
            if self._image_key is not None:
                c, h, w = image_chw
                if self._image_is_resnet:
                    cnn_out = 512
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
            if self._image_key is not None:
                if self._image_is_resnet:
                    pretrained = bool((cnn_params or {}).get("pretrained", True))
                    weights_path = (cnn_params or {}).get("weights_path")
                    self._image_cnn = self._build_resnet18(image_chw[0], pretrained, weights_path)
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
        def _build_resnet18(in_channels, pretrained, weights_path=None):
            """ImageNet ResNet-18 encoder, input conv inflated to ``in_channels``, backbone FROZEN.

            - conv1 (3ch) -> (in_channels)ch: copy the RGB weights; init each extra (depth) channel to the
              MEAN of the RGB conv weights (a sane grey-world init). conv1 is kept TRAINABLE so the depth
              channel adapts; every other backbone param is frozen.
            - fc is replaced by Identity so forward() returns the 512-d global-average-pooled feature.
            - BatchNorm stays in eval mode at forward time (see forward()) so the frozen running stats are
              used, not noisy RL-minibatch stats. Robust to being offline: if the pretrained download fails
              we fall back to random init (still trainable conv1) with a warning; an optional local
              ``weights_path`` (a torchvision resnet18 state_dict) is loaded first if given.
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

            old_conv = net.conv1  # Conv2d(3, 64, k=7, s=2, p=3, bias=False)
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                w = old_conv.weight  # (64, 3, 7, 7)
                if in_channels >= 3:
                    new_conv.weight[:, :3] = w
                    if in_channels > 3:
                        new_conv.weight[:, 3:] = w.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
                else:
                    new_conv.weight[:] = w[:, :in_channels]
            net.conv1 = new_conv

            for p in net.parameters():
                p.requires_grad = False
            for p in net.conv1.parameters():  # keep ONLY the (inflated) input conv trainable
                p.requires_grad = True
            net.fc = nn.Identity()  # forward() returns the 512-d pooled feature
            return net

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            # Concatenate proprio groups (everything that is neither the image nor the privileged aux
            # label), preserving dict order.
            proprio = [v for k, v in obs.items() if k != self._image_key and k != self._aux_label_key]
            feat = torch.cat(proprio, dim=1) if proprio else None
            self._aux_loss = None
            self._estimator_pred = None  # last explicit-estimator prediction (for viz/logging)
            if self._image_cnn is not None:
                img = obs[self._image_key].permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
                if self._image_is_resnet:
                    # Keep the frozen backbone's BatchNorm on its ImageNet running stats regardless of the
                    # agent toggling model.train()/eval(); the trainable conv1 still gets gradients (eval
                    # mode gates BN/dropout, not autograd).
                    self._image_cnn.eval()
                cnn_feat = self._image_proj(self._image_cnn(img).flatten(1))  # cnn_out -> proj_dim
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
            return super().forward(fused)

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
