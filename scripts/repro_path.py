#!/usr/bin/env python3
"""Which in-process forward PATH turns the frozen resnet's healthy ~6.5 into 1e31?

Correct ImageNet weights (proven == in-run backbone_state) + the exact fwd1 frame. Eager fp32 is healthy;
test the paths the training process actually uses: nested fp16-outer+bf16-inner autocast, and torch.compile.
"""
import sys

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def am(t):
    t = t.detach().float()
    fin = torch.isfinite(t)
    return float(t[fin].abs().max()) if bool(fin.any()) else float("inf")


def ac(net, x, dt):
    with torch.autocast("cuda", dtype=dt):
        return net(x)


def nested(net, x):  # rl_games mixed_precision(fp16) OUTER, our encoder bf16 INNER
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(x)


def run(tag, fn):
    with torch.no_grad():
        out = fn().flatten(1)
    print(f"  {tag:34s} out_absmax={am(out):.4g}{'   <== REPRODUCES' if am(out) > 1e4 else ''}")


d = torch.load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/resnet_nan_frame_fwd1.pt", map_location="cpu")
x = d["img_bchw"].cuda()
net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # proven identical to in-run backbone_state
net.fc = nn.Identity()
net.eval().cuda()
for p in net.parameters():
    p.requires_grad = False
print(f"frame {tuple(x.shape)} absmax={am(x):.3g} | in-run resnet_out was {am(d['resnet_out']):.3g}")

run("eager fp32", lambda: net(x))
run("autocast(fp16)", lambda: ac(net, x, torch.float16))
run("autocast(bf16)", lambda: ac(net, x, torch.bfloat16))
run("NESTED fp16-out + bf16-in", lambda: nested(net, x))
try:
    cnet = torch.compile(net, mode="default")
    run("compiled fp32", lambda: cnet(x))
    run("compiled autocast(fp16)", lambda: ac(cnet, x, torch.float16))
    run("compiled NESTED fp16+bf16", lambda: nested(cnet, x))
except Exception as e:
    print(f"  torch.compile failed: {type(e).__name__}: {e}")
