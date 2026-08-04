"""Progress plot for the E2E obs-restoration experiment: 4-panel (default) or 9-panel (--full).

STITCHES every event file per run (auto-resume writes several -> a naive events[-1] load only shows the
last segment and makes resumed runs look truncated). All panels are plotted against ITERATION (rl_games
logs info/* and losses/* against the FRAME count, so we use the per-epoch index instead). CPU-only, safe
to run while the GPU trains.

    python scripts/plot_e2e_progress.py                    # 4 core panels
    python scripts/plot_e2e_progress.py --full             # 9 panels (reward/success/tip/axis/kl/lr/a/c/ent)
    python scripts/plot_e2e_progress.py --full --runs a b --out /tmp/x.png
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "logs/rl_games/Forge"
TAGS = {
    "success": "successes/iter",
    "reward": "rewards/iter",
    "tip_dist": "logs_rew_neg_tip_l2/iter",       # logged as -tip_dist -> negate to a positive distance
    "axis_err": "logs_rew_shaft_axis_error/iter",
    "kl": "info/kl",
    "lr": "info/last_lr",
    "a_loss": "losses/a_loss",
    "c_loss": "losses/c_loss",
    "entropy": "losses/entropy",
}
DEFAULT = ["vision_res224_1", "e2e_vis_proprio",
           "e2e_vis_tilt08", "e2e_vis_tilt16", "e2e_vis_tilt20", "e2e_vis_tilt24"]

# panel := (key, title, ylabel, logy, transform)
NEG = lambda y: -y
ID = lambda y: y
CORE = [("success", "training SUCCESS", "fraction", False, ID),
        ("reward", "episode REWARD", "reward", False, ID),
        ("kl", "policy KL (info/kl)", "KL", True, ID),
        ("lr", "LEARNING RATE (info/last_lr)", "lr", True, ID)]
FULL = [("reward", "episode REWARD", "reward", False, ID),
        ("success", "training SUCCESS", "fraction", False, ID),
        ("tip_dist", "TIP DISTANCE to socket", "metres", False, NEG),
        ("axis_err", "shaft AXIS ERROR", "rad", False, ID),
        ("kl", "policy KL (info/kl)", "KL", True, ID),
        ("lr", "LEARNING RATE (info/last_lr)", "lr", True, ID),
        ("a_loss", "ACTOR loss (losses/a_loss)", "loss", False, ID),
        ("c_loss", "CRITIC loss (losses/c_loss)", "loss", False, ID),
        ("entropy", "ENTROPY (losses/entropy)", "entropy", False, ID)]


def load_stitched(run):
    evs = sorted(glob.glob(os.path.join(ROOT, run, "summaries", "events*")), key=os.path.getmtime)
    series = {k: {} for k in TAGS}
    for ev in evs:
        ea = EventAccumulator(ev, size_guidance={"scalars": 0}); ea.Reload()
        avail = ea.Tags().get("scalars", [])
        for k, t in TAGS.items():
            if t in avail:
                for s in ea.Scalars(t):
                    series[k][s.step] = s.value  # dedup by step, keep last-written
    return {k: np.array([d[x] for x in sorted(d)]) for k, d in series.items() if d}


def smooth(y, w=15):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT)
    ap.add_argument("--full", action="store_true", help="9-panel diagnostic instead of the 4 core panels")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth", type=int, default=15)
    args = ap.parse_args()

    data = {r: load_stitched(r) for r in args.runs if os.path.isdir(os.path.join(ROOT, r))}
    runs = list(data)
    hi = {"vision_res224_1": ("black", 3.0), "e2e_vis_proprio": ("tab:green", 3.0)}
    faded = plt.cm.autumn(np.linspace(0.15, 0.75, max(1, len(runs))))
    colors, fi = {}, 0
    for r in runs:
        colors[r] = hi[r] if r in hi else (faded[fi], 1.4)
        fi += 0 if r in hi else 1

    print("=== stitched span + endpoint (last-50 mean) ===")
    for r in runs:
        d = data[r]
        def last(k):
            return d[k][-50:].mean() if k in d and len(d[k]) else float("nan")
        n = len(d.get("success", []))
        print(f"  {r:16s} iters ~{n:4d}  succ~{last('success'):.3f} rew~{last('reward'):.1f} "
              f"tipD~{-last('tip_dist'):.4f} axisErr~{last('axis_err'):.3f} kl~{last('kl'):.4f} "
              f"ent~{last('entropy'):.2f}")

    panels = FULL if args.full else CORE
    nrow, ncol = (3, 3) if args.full else (2, 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 14) if args.full else (17, 11))
    for ax, (key, title, ylab, logy, tf) in zip(axes.flat, panels):
        for r in runs:
            if key not in data[r]:
                continue
            y = tf(data[r][key]); ys = smooth(y, args.smooth)
            it = np.arange(len(y))
            xs = it[args.smooth - 1:] if len(y) >= args.smooth else it
            c, lw = colors[r]
            ax.plot(xs, ys, label=r, color=c, lw=lw)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("iteration"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
        if key == "kl":
            ax.axhline(0.008, ls="--", c="red", lw=1.1, alpha=0.7)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("E2E obs restoration: e2e_vis_proprio (green) vs res224 (black) vs weekend tilt sweep",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or ("diagnostics/e2e_proprio_full.png" if args.full else "diagnostics/e2e_proprio_vs.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
