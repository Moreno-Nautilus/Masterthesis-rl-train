"""Overlay training curves of two+ rl_games runs for A/B comparison (e.g. aux vs no-aux).

CPU-only (TensorBoard event files), safe to run while the GPU trains. Panels: training success and
reward per iteration (overlaid across runs), plus any auxiliary-head losses (grasp/hole) for runs
that have them (their frame-indexed steps are converted to iterations).

    python scripts/plot_compare.py --runs vision_appearance_1 vision_auxhead_1 --out /tmp/aux_vs_noaux.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "logs/rl_games/Forge"


def load(run):
    ea = EventAccumulator(os.path.join(ROOT, run, "summaries"), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    s = ea.Scalars(tag)
    return np.array([p.step for p in s], dtype=float), np.array([p.value for p in s], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under logs/rl_games/Forge/")
    ap.add_argument("--out", default="renders/analysis/run_compare.png")
    ap.add_argument("--smooth", type=int, default=25, help="moving-average window (iters) for the noisy curves")
    args = ap.parse_args()

    eas = {r: load(r) for r in args.runs}

    def smooth(y):
        if y is None or args.smooth <= 1:
            return y
        k = min(args.smooth, len(y))
        return np.convolve(y, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1+2: success + reward per ITERATION (overlaid)
    for tag, ax, title in [("successes/iter", axes[0], "training success"),
                           ("rewards/iter", axes[1], "reward")]:
        for r in args.runs:
            x, y = series(eas[r], tag)
            if y is None:
                continue
            ys = smooth(y)
            xs = x[len(x) - len(ys):]
            ax.plot(xs, ys, label=r, lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Panel 3: aux losses (grasp/hole), frames -> iterations
    ax = axes[2]
    any_aux = False
    for r in args.runs:
        # infer samples/iter = max(loss step) / max(iter step)
        xs_iter, _ = series(eas[r], "rewards/iter")
        spi = 8192.0
        for nm, col in [("losses/grasp", None), ("losses/hole", None)]:
            xf, y = series(eas[r], nm)
            if y is None:
                continue
            any_aux = True
            if xs_iter is not None and xf.max() > 0:
                spi = xf.max() / xs_iter.max()
            ax.plot(xf / spi, y, lw=1.2, label=f"{r}:{nm.split('/')[1]}")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8, label="predict-mean baseline")
    ax.set_title("aux losses (normalized MSE)")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)
    if any_aux:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no aux losses in these runs", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(" vs ".join(args.runs), fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
