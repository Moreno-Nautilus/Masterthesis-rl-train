#!/usr/bin/env python3
"""Replay a dumped triggering frame against a BARE frozen resnet18 to isolate the NaN explosion.

Context (2026-08-20 NaN debug): the frozen ImageNet ResNet-18 in insertion_hybrid_network intermittently
emits ~1e36 features on a specific rendered frame -> fp16 overflow -> NaN actor mu. The training run dumps
the exact offending frame to /tmp/resnet_nan_frame_fwd<N>.pt (see insertion_hybrid_network.forward,
DEBUG_NAN block). This script loads that frame and:

  1. Feeds it through a bare torchvision resnet18 (ImageNet weights, fc=Identity, eval) in fp32/bf16/fp16
     to confirm the explosion reproduces IN ISOLATION (rules the RL stack in or out).
  2. If it reproduces, registers per-module forward hooks and prints each layer's output absmax so the
     FIRST layer that blows up is visible -- typically a BN channel with a tiny running_var hit by this
     specific frame.
  3. Prints, for that layer's BN, the channel with min running_var and the input stats feeding it.

Usage:
    python scripts/repro_resnet_nan.py /tmp/resnet_nan_frame_fwd123.pt
    python scripts/repro_resnet_nan.py            # auto-picks the newest /tmp/resnet_nan_frame_fwd*.pt
"""
import glob
import sys

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _absmax(t):
    t = t.detach().float()
    fin = torch.isfinite(t)
    return float(t[fin].abs().max()) if bool(fin.any()) else float("inf")


def _pick_dump(argv):
    if len(argv) > 1:
        return argv[1]
    cands = sorted(glob.glob("/tmp/resnet_nan_frame_fwd*.pt"))
    if not cands:
        sys.exit("no /tmp/resnet_nan_frame_fwd*.pt found -- run training with DEBUG_NAN=1 first")
    return cands[-1]


def main():
    path = _pick_dump(sys.argv)
    print(f"[repro] loading {path}")
    dump = torch.load(path, map_location="cpu")
    img = dump["img_bchw"]  # (B,C,H,W), fp32, exactly as fed to the backbone in-run
    bad = dump.get("bad_rows", torch.arange(img.shape[0]))
    print(f"[repro] img shape={tuple(img.shape)} dtype={img.dtype} "
          f"absmax={_absmax(img):.4g} min={float(img.min()):.4g} max={float(img.max()):.4g}")
    print(f"[repro] in-run resnet_out absmax={_absmax(dump['resnet_out']):.4g} "
          f"bad_rows={bad.tolist()}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval().to(dev)
    for p in net.parameters():
        p.requires_grad = False

    # Focus on the offending rows if we have them; else use the whole batch.
    rows = bad if bad.numel() > 0 else torch.arange(img.shape[0])
    x = img[rows].to(dev)

    print("\n[repro] === bare resnet18 output absmax by dtype (offending rows) ===")
    reproduced = False
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        with torch.no_grad(), torch.autocast("cuda", dtype=dt, enabled=(dev == "cuda")):
            out = net(x.to(dt) if dev == "cpu" else x).flatten(1)
        am = _absmax(out)
        flag = "  <-- EXPLODES" if (am > 1e4 or not torch.isfinite(out).all()) else ""
        print(f"    {str(dt):>16}: out_absmax={am:.4g}{flag}")
        if am > 1e4:
            reproduced = True

    # --- Load the TRAINED backbone/conv1 (if dumped) and test the same frames. This isolates whether the
    #     trained conv1 (the only trainable backbone param) is the amplifier vs the frozen ImageNet weights.
    if "conv1_weight" in dump or "backbone_state" in dump:
        c0 = net.conv1.weight.detach().float()
        c1 = dump.get("conv1_weight")
        if c1 is not None:
            c1 = c1.float()
            print(f"\n[repro] conv1 weight drift: ImageNet absmax={float(c0.abs().max()):.4g} fro={float(c0.norm()):.4g}"
                  f"  |  TRAINED absmax={float(c1.abs().max()):.4g} fro={float(c1.norm()):.4g}"
                  f"  |  delta_fro={float((c1 - c0).norm()):.4g} max_delta={float((c1 - c0).abs().max()):.4g}")
        trained = resnet18(weights=None)
        trained.fc = nn.Identity()
        if "backbone_state" in dump:
            trained.load_state_dict(dump["backbone_state"], strict=False)
        elif c1 is not None:
            trained.load_state_dict(dict(net.state_dict()), strict=False)  # ImageNet base
            trained.conv1.weight.data.copy_(c1)                            # overlay trained conv1
        trained.eval().to(dev)
        with torch.no_grad():
            out_t = trained(x.float()).flatten(1)
        am_t = _absmax(out_t)
        print(f"[repro] TRAINED backbone on same frames: out_absmax={am_t:.4g}"
              f"{'  <-- EXPLODES => conv1/backbone drift is the cause' if am_t > 1e4 else '  (healthy)'}")

    if not reproduced:
        print("\n[repro] Bare (ImageNet) resnet18 did NOT reproduce on this frame in fp32.")
        print("        => the trigger is NOT input content. If the TRAINED backbone above EXPLODES, the")
        print("           trainable conv1 is the amplifier (fix: freeze conv1 for RGB-only, or normalize")
        print("           after the frozen backbone / lower conv1 LR).")
        return

    print("\n[repro] REPRODUCED. Per-module output absmax (fp32) -- first layer >1e3 is the culprit:")
    hooks = []
    log = []

    def mk(name):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            log.append((name, _absmax(o), _absmax(_inp[0]) if _inp else float("nan")))
        return hook

    for name, m in net.named_modules():
        if name and len(list(m.children())) == 0:  # leaf modules only
            hooks.append(m.register_forward_hook(mk(name)))
    with torch.no_grad():
        net(x.float())
    for h in hooks:
        h.remove()

    prev = 0.0
    culprit = None
    for name, out_am, in_am in log:
        jump = "  <== JUMP" if (out_am > 1e3 and prev <= 1e3) else ""
        print(f"    {name:<28} in={in_am:>11.4g} out={out_am:>11.4g}{jump}")
        if culprit is None and out_am > 1e3 and prev <= 1e3:
            culprit = name
        prev = out_am

    if culprit:
        print(f"\n[repro] First exploding layer: {culprit}")
        mod = dict(net.named_modules())[culprit]
        if isinstance(mod, nn.BatchNorm2d):
            rv = mod.running_var
            i = int(torch.argmin(rv))
            print(f"    BatchNorm2d: min running_var={float(rv.min()):.4g} @ch{i} "
                  f"(running_mean@ch={float(mod.running_mean[i]):.4g}, "
                  f"gamma@ch={float(mod.weight[i]):.4g}, beta@ch={float(mod.bias[i]):.4g})")
            print("    => a near-zero running_var channel divides this frame's activation to ~1e18;")
            print("       fixes: LayerNorm/BN after the frozen backbone, feature clamp, or unfreeze layer4.")


if __name__ == "__main__":
    main()
