"""Episode-reward curves for the arms-locked sprint, one run per symmetry method.

Reads the TensorBoard summaries of iteration 2026-09-20_v1 and writes
reward_curves_en.pdf / .png. English axis labels; each panel is titled
with the method.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
LAST_ITERATION = 10000
RUNS = (
    ("BASE", "BASE/logs/2026-09-20-21-01-54/summaries", "#0072B2"),
    ("DUP", "dup/logs/2026-09-21-12-02-13/summaries", "#E69F00"),
    ("LOSS", "loss/logs/2026-09-22-03-11-16/summaries", "#009E73"),
    ("PHASE", "phase/logs/2026-10-20-05-26-56/summaries", "#D55E00"),
)


def load_reward(rel):
    acc = EventAccumulator(os.path.join(HERE, rel), size_guidance={SCALARS: 0})
    acc.Reload()
    events = acc.Scalars("reward")
    steps = np.asarray([event.step for event in events], dtype=np.int64)
    values = np.asarray([event.value for event in events], dtype=np.float64)
    return steps, values


def main():
    loaded = [(name, color, *load_reward(rel)) for name, rel, color in RUNS]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), sharex=True, sharey=True)
    for ax, (name, color, steps, reward) in zip(axes.flat, loaded):
        keep = steps <= LAST_ITERATION
        steps, reward = steps[keep], reward[keep]
        ax.plot(steps, reward, color=color, lw=0.8)
        ax.set_title(name, loc="left")
        ax.set_xlim(0, LAST_ITERATION)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0)
        ax.tick_params(labelsize=10)
        print("{} first/last/max: {:.4g} / {:.4g} / {:.4g}".format(name, reward[0], reward[-1], reward.max()))

    axes[1, 0].set_xlabel("Training iteration")
    axes[1, 1].set_xlabel("Training iteration")
    axes[0, 0].set_ylabel("Episode reward")
    axes[1, 0].set_ylabel("Episode reward")
    fig.tight_layout()

    pdf_path = os.path.join(HERE, "reward_curves_en.pdf")
    png_path = os.path.join(HERE, "reward_curves_en.png")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    print("wrote", png_path)
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        print("wrote", pdf_path)
    except PermissionError:
        alt = os.path.join(HERE, "reward_curves_en_to10000.pdf")
        fig.savefig(alt, bbox_inches="tight")
        print("pdf locked, wrote", alt)


if __name__ == "__main__":
    main()
