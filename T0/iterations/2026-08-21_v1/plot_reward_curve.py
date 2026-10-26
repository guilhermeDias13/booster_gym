"""Episode-reward curve for T0 run logs/2026-08-21-14-44-42.

Reads the TensorBoard summaries and writes reward_curve_en.pdf / .png
with English axis labels.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARIES = os.path.join(HERE, "logs", "2026-08-21-14-44-42", "summaries")


def main():
    acc = EventAccumulator(SUMMARIES, size_guidance={SCALARS: 0})
    acc.Reload()
    reward_events = acc.Scalars("reward")
    iterations = np.asarray([event.step for event in reward_events], dtype=np.int64)
    reward = np.asarray([event.value for event in reward_events], dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.plot(iterations, reward, color="#0072B2", lw=0.9)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Episode reward")
    ax.set_xlim(iterations[0], iterations[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0)

    pdf_path = os.path.join(HERE, "reward_curve_en.pdf")
    png_path = os.path.join(HERE, "reward_curve_en.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print("wrote", pdf_path)
    print("wrote", png_path)
    print("return first/last/max: {:.4g} / {:.4g} / {:.4g}".format(reward[0], reward[-1], reward.max()))


if __name__ == "__main__":
    main()
