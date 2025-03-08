import matplotlib.pyplot as plt
from adjustText import adjust_text

import numpy as np


HPA_MAP = {
    "Full FT": [50.57, 104.76],
    "LoRA": [49.31, 16.27],
    "Freeze Encoder": [37.38, 15.09],
    "Late-LoRA (C-50%)": [43.93, 15.68],
    "Late-LoRA (A-50%)": [43.66, 17.45],
    "Late-LoRA (C-25%)": [40.98, 15.29],
    "Late-LoRA (A-25%)": [40.66, 16.27],
    "Late-LoRA (C-8%)": [40.26, 15.19],
    "Late-LoRA (A-8%)": [40.45, 15.49],
    "Late-FT (50%)": [43.67, 57.67],
    "Late-FT (25%)": [40.92, 36.38],
    "Late-FT (8%)": [40.52, 22.2],
}

PSFHS_MAP = {
    "Full FT": [20.49, 93.74],
    "LoRA": [19.59, 4.06],
    "Freeze Encoder": [7.64, 5.24],
    "Late-LoRA (C-50%)": [14.22, 4.65],
    "Late-LoRA (A-50%)": [13.9, 6.42],
    "Late-LoRA (C-25%)": [10.54, 4.36],
    "Late-LoRA (A-25%)": [10.65, 5.24],
    "Late-LoRA (C-8%)": [9.71, 4.16],
    "Late-LoRA (A-8%)": [9.79, 4.46],
    "Late-FT (50%)": [13.87, 46.64],
    "Late-FT (25%)": [10.64, 25.35],
    "Late-FT (8%)": [9.81, 11.17],
}


def _plot_param_vs_mem(dataset):
    if dataset == "hpa":
        maps = HPA_MAP
        mname = r"$\boldsymbol{\mu}$SAM"
    else:
        maps = PSFHS_MAP
        mname = "MedicoSAM"

    num_points = len(maps)
    cmap = plt.get_cmap('tab20b', num_points)
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, num_points)]

    plt.figure(figsize=(10, 6))

    texts = []
    non_text_points = []

    for (label, (memory, params)), color in zip(maps.items(), colors):
        if "Late" in label:
            non_text_points.append((memory, params, label, color))
            plt.scatter(memory, params, color=color, alpha=0.7)
        else:
            plt.scatter(memory, params, color=color, alpha=0.7)

            # Adjusting label positioning
            # NOTE: I will post-process them anyways. Doesn't matter!
            if label == "Full FT":
                text_x, text_y = memory, params - 2
                ha, va = 'center', 'top'
            else:
                text_x, text_y = memory, params + 1.5
                ha, va = 'left', 'bottom'

            text = plt.text(
                text_x, text_y, label, fontsize=9, ha=ha, va=va, color=color,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.3')
            )
            texts.append(text)

    adjust_text(texts, arrowprops=None)

    if non_text_points:
        x_values, y_values = zip(*[(x, y) for x, y, _, _ in non_text_points])
        min_x, max_x = min(x_values) - 0.5, max(x_values) + 0.5
        min_y, max_y = min(y_values) - 2, max(y_values) + 2
        plt.gca().add_patch(
            plt.Rectangle(
                (min_x, min_y), max_x - min_x, max_y - min_y, fill=True, facecolor='lightgray',
                alpha=0.3, edgecolor='black', linestyle='dotted', linewidth=1.5
            )
        )

        plt.text(
            (min_x + max_x) / 2, max_y + 1, "Late PEFT Methods", fontsize=12,
            ha='center', va='bottom', color='black', fontstyle='italic'
        )

    plt.xlabel("Memory (in GB)")
    plt.ylabel("Parameter Count (in Million)")
    plt.title(f"Parameter vs. Memory Combination (PEFT Methods on {mname})", fontsize=12, fontweight="bold")
    plt.savefig(f"./full_plot_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./full_plot_{dataset}.svg", bbox_inches='tight')

    plt.figure(figsize=(8, 5))

    zoom_texts = []
    for memory, params, label, color in non_text_points:
        plt.scatter(memory, params, color=color, alpha=0.7)

        # NOTE: Same. I will post-process it.
        if label == "Late-FT (50%)":
            text_x, text_y = memory, params - 1.5
            ha, va = 'center', 'top'
        elif label in ["Late-LoRA (A-1%)", "Late-LoRA (C-1%)"]:
            text_x, text_y = memory, params + 1
            ha, va = 'center', 'bottom'
        elif label == "Late-LoRA (C-50%)":
            text_x, text_y = memory - 0.25, params + 0.2
            ha, va = 'right', 'center'
        else:
            text_x, text_y = memory + 0.2, params + 1.2

        text = plt.text(
            text_x, text_y, label, fontsize=9, ha=ha, va=va, color=color,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.3')
        )
        zoom_texts.append(text)

    adjust_text(zoom_texts, arrowprops=None)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel("Memory (in GB)")
    plt.ylabel("Parameter Count (in Million)")
    plt.title("Late PEFT Methods", fontsize=12, fontweight="bold")
    plt.savefig(f"./zoomed_plot_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./zoomed_plot_{dataset}.svg", bbox_inches='tight')


_plot_param_vs_mem("hpa")
_plot_param_vs_mem("psfhs")
