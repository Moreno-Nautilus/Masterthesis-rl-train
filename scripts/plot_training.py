"""Render the key training curves from an rl_games TensorBoard event file to a PNG.

CPU-only (no Isaac Sim), so it is safe to run while a training job is using the GPU. Refresh the
plot anytime to see the latest trend:

    python scripts/plot_training.py [--logdir logs/rl_games/Forge/<run>] [--out <path.png>]
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    s = ea.Scalars(tag)
    return np.array([p.step for p in s]), np.array([p.value for p in s])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logs/rl_games/Forge/tilt_learnability")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = sorted(glob.glob(os.path.join(args.logdir, "summaries", "events.*")))
    if not events:
        raise SystemExit(f"no event files under {args.logdir}/summaries")
    ea = EventAccumulator(events[-1], size_guidance={"scalars": 0})
    ea.Reload()

    # (tag, title, y-label, transform)
    panels = [
        ("rewards/iter", "episode reward", "reward", lambda y: y),
        ("successes/iter", "success rate", "fraction", lambda y: y),
        ("logs_rew_shaft_axis_error/iter", "mean shaft-axis error", "deg", np.degrees),
        ("logs_rew_neg_keypoint_l2/iter", "neg keypoint L2 (closeness)", "m", lambda y: y),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (tag, title, ylab, fn) in zip(axes.ravel(), panels):
        x, y = _series(ea, tag)
        if x is None:
            ax.set_title(f"{title}\n(missing: {tag})")
            continue
        yv = fn(y)
        ax.plot(x, yv, lw=0.8, alpha=0.35, color="C0")  # raw (jittery)
        w = max(1, len(yv) // 20)  # rolling-mean trend so the signal shows through the noise
        if w > 1:
            kern = np.ones(w) / w
            sm = np.convolve(yv, kern, mode="valid")
            ax.plot(x[w - 1 :], sm, lw=2.2, color="C3", label=f"{w}-epoch mean")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{title}  (latest={yv[-1]:.3f}, n={len(yv)})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    fig.suptitle(os.path.basename(args.logdir.rstrip("/")), fontweight="bold")
    fig.tight_layout()

    out = args.out or os.path.join(args.logdir, "progress.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  (from {os.path.basename(events[-1])})")


if __name__ == "__main__":
    main()
