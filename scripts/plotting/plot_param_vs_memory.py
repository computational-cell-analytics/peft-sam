import matplotlib.pyplot as plt
from adjustText import adjust_text

import numpy as np


HPA_MAP = {
    "Full FT": [56.87, 104.76],
    "LoRA": [56.78, 16.27],
    "Freeze Encoder": [45.5, 15.09],
    "Late-LoRA (C-50%)": [56.93, 15.68],
    "Late-LoRA (A-50%)": [51.75, 17.45],
    "Late-LoRA (C-25%)": [52.52, 15.29],
    "Late-LoRA (A-25%)": [53.08, 16.27],
    "Late-LoRA (C-1%)": [51.25, 15.19],
    "Late-LoRA (A-1%)": [51.40, 15.49],
    "Late-FT (50%)": [52.08, 57.67],
    "Late-FT (25%)": [52.58, 36.38],
    "Late-FT (1%)": [51.31, 22.2],
}

PSFHS_MAP = {
    "Full FT": [22, 93.74],
    "LoRA": [20, 4.06],
    "Freeze Encoder": [7, 5.24],
    "Late-LoRA (C-50%)": [14.34, 4.65],
    "Late-LoRA (A-50%)": [13.9, 6.42],
    "Late-LoRA (C-25%)": [10.5, 4.36],
    "Late-LoRA (A-25%)": [10.52, 5.24],
    "Late-LoRA (C-1%)": [9.5, 4.16],
    "Late-LoRA (A-1%)": [9.57, 4.46],
    "Late-FT (50%)": [13.97, 46.64],
    "Late-FT (25%)": [10.5, 25.35],
    "Late-FT (1%)": [9.59, 11.17],
}


def _plot_param_vs_mem(dataset):
    if dataset == "hpa":
        maps = HPA_MAP
        mname = r"$\boldsymbol{\mu}$SAM"  # Bold μ in LaTeX
    else:
        maps = PSFHS_MAP
        mname = "MedicoSAM"

    num_points = len(maps)  # Number of data points
    cmap = plt.get_cmap('tab20b', num_points)
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, num_points)]  # Extract only RGB (avoid alpha transparency issues)

    plt.figure(figsize=(10, 6))

    texts = []
    non_text_points = []

    # First plot: Full plot (without text for "Late" points)
    for (label, (memory, params)), color in zip(maps.items(), colors):
        if "Late" in label:
            # Collect points for zoomed-in plot
            non_text_points.append((memory, params, label, color))
            plt.scatter(memory, params, color=color, alpha=0.7)  # No black border
        else:
            plt.scatter(memory, params, color=color, alpha=0.7)

            # Adjust label positioning
            if label == "Full FT":
                text_x, text_y = memory, params - 2  # Keep close, but below
                ha, va = 'center', 'top'
            else:
                text_x, text_y = memory, params + 1.5  # Keep close, but above
                ha, va = 'left', 'bottom'

            text = plt.text(text_x, text_y, label, fontsize=9, ha=ha, va=va, color=color,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.3'))
            texts.append(text)

    # Adjust text positions to prevent overlap (NO ARROWS)
    adjust_text(texts, arrowprops=None)

    # Draw a light gray shaded region with a dotted black border
    if non_text_points:
        x_values, y_values = zip(*[(x, y) for x, y, _, _ in non_text_points])
        min_x, max_x = min(x_values) - 0.5, max(x_values) + 0.5
        min_y, max_y = min(y_values) - 2, max(y_values) + 2
        plt.gca().add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                          fill=True, facecolor='lightgray', alpha=0.3, edgecolor='black',
                                          linestyle='dotted', linewidth=1.5))

        # **Add title above the gray square in the same gray color as the border**
        plt.text((min_x + max_x) / 2, max_y + 1, "Late PEFT Methods", fontsize=12,
                 ha='center', va='bottom', color='black', fontstyle='italic')

    # Save full plot
    plt.xlabel("Memory (in GB)")
    plt.ylabel("Parameter Count (in Million)")
    plt.title(f"Parameter vs. Memory Combination (PEFT Methods on {mname})", fontsize=12, fontweight="bold")
    plt.savefig("./full_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig("./full_plot.svg", bbox_inches='tight')

    # Create zoomed-in plot
    plt.figure(figsize=(8, 5))

    zoom_texts = []
    for memory, params, label, color in non_text_points:
        plt.scatter(memory, params, color=color, alpha=0.7)

        # Custom placement for specific labels
        if label == "Late-FT (50%)":
            text_x, text_y = memory, params - 1.5  # Place label below
            ha, va = 'center', 'top'
        elif label in ["Late-LoRA (A-1%)", "Late-LoRA (C-1%)"]:
            text_x, text_y = memory, params + 1  # Place label above
            ha, va = 'center', 'bottom'
        elif label == "Late-LoRA (C-50%)":
            text_x, text_y = memory - 0.25, params + 0.2  # Move label to the left
            ha, va = 'right', 'center'
        else:
            text_x, text_y = memory + 0.2, params + 1.2  # Default: slight offset

        text = plt.text(
            text_x, text_y, label, fontsize=9, ha=ha, va=va, color=color,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.3')
        )
        zoom_texts.append(text)

    # Adjust text positions in zoomed-in plot (NO ARROWS)
    adjust_text(zoom_texts, arrowprops=None)

    # Set limits based on the previous gray box
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Save zoomed-in plot
    plt.xlabel("Memory (in GB)")
    plt.ylabel("Parameter Count (in Million)")
    plt.title("Late PEFT Methods", fontsize=12, fontweight="bold")
    plt.savefig("./zoomed_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig("./zoomed_plot.svg", bbox_inches='tight')


_plot_param_vs_mem("hpa")
