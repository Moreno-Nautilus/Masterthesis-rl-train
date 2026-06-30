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

            # Measure the CNN's flattened output with a throwaway module (cannot store an nn.Module
            # before nn.Module.__init__ runs inside the parent constructor).
            cnn_out = 0
            if self._image_key is not None:
                c, h, w = image_chw
                with torch.no_grad():
                    cnn_out = self._make_cnn(c, convs)(torch.zeros(1, c, h, w)).flatten(1).shape[1]

            # Project the (large) CNN feature vector down before fusing with proprio, so proprio is
            # not drowned. With no image (the critic) there is no projection. Plain ints only here.
            self._cnn_out = cnn_out
            self._proj_dim = int((cnn_params or {}).get("fc_size", 128)) if self._image_key is not None else 0

            # Build the rest of the net as a flat (proj + proprio)-vector net: strip 'cnn' so the
            # parent does not build its own CNN, and pass the fused feature dim as the input shape.
            parent_params = copy.deepcopy(params)
            parent_params.pop("cnn", None)
            parent_params.pop("aux_head", None)
            parent_kwargs = dict(kwargs)
            parent_kwargs["input_shape"] = (self._proj_dim + proprio_dim,)
            super().__init__(parent_params, **parent_kwargs)

            # Now that nn.Module is initialized, build the real CNN + projection FC that forward() uses.
            if self._image_key is not None:
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
                heads = {}
                for name, spec in (aux_cfg.get("targets") or {}).items():
                    lo, hi = int(spec["slice"][0]), int(spec["slice"][1])
                    heads[name] = nn.Sequential(
                        nn.Linear(self._proj_dim, hidden), nn.ReLU(), nn.Linear(hidden, hi - lo)
                    )
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

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            # Concatenate proprio groups (everything that is neither the image nor the privileged aux
            # label), preserving dict order.
            proprio = [v for k, v in obs.items() if k != self._image_key and k != self._aux_label_key]
            feat = torch.cat(proprio, dim=1) if proprio else None
            self._aux_loss = None
            if self._image_cnn is not None:
                img = obs[self._image_key].permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
                cnn_feat = self._image_proj(self._image_cnn(img).flatten(1))  # 3136 -> proj_dim
                # Compute the aux loss only on training forwards (skipped during rollout/eval, where the
                # heads are unused) and only when the privileged label is present.
                if self.training and self._aux_heads is not None and self._aux_label_key in obs:
                    self._aux_loss = self._compute_aux_loss(cnn_feat, obs[self._aux_label_key])
                feat = cnn_feat if feat is None else torch.cat([cnn_feat, feat], dim=1)
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
