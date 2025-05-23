import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from glob import glob
import os
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

all_datasets = ['psfhs', 'hpa']
methods = ['lora', 'ClassicalSurgery']
update_matrices = ['standard', 'all_matrices']
attention_layers_to_update = [[6, 7, 8, 9, 10, 11], [9, 10, 11], [11]]

CUSTOM_PALETTE = {
    "ais": "#045275",
    "point": "#7CCBA2",
    "box": "#90477F",
    "ip": "#089099",
    "ib": "#F0746E",
}

DATASETS = {
    "amd_sd": "AMD-SD",
    "jsrt": "JSRT",
    "mice_tumseg": "Mice TumSeg",
    "papila": "Papila",
    "motum": "MOTUM",
    "psfhs": "PSFHS",
    "dsad": "DSAD",
    "sega": "SEGA",
    "ircadb": "IRCADb",
    "covid_if": "Covid-IF",
    "orgasegment": "OrgaSegment",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "mitolab_glycolytic_muscle": "MitoLab",
    "platy_cilia": "Platynereis",
}

DOMAIN = {
    "amd_sd": "medical",
    "jsrt": "medical",
    "mice_tumseg": "medical",
    "papila": "medical",
    "motum": "medical",
    "psfhs": "medical",
    "dsad": "medical",
    "sega": "medical",
    "ircadb": "medical",
    "covid_if": "microscopy",
    "orgasegment": "microscopy",
    "gonuclear": "microscopy",
    "hpa": "microscopy",
    "mitolab_glycolytic_muscle": "microscopy",
    "platy_cilia": "microscopy",
}


def plot_late_lora(data, domain="medical"):
    # metrics = ['box', 'point', 'ib', 'ip', 'ais']
    metrics = ['box', 'ais', 'point']
    df_long = pd.melt(
        data,
        id_vars=["dataset", "method", "update_matrices", "start_layer"],
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    )

    replace_layer = {0: '100%', 6: '50%', 9: '25%', 11: '8%', 12: 'Freeze Encoder (0%)'}
    df_long['start_layer'] = df_long['start_layer'].replace(replace_layer)
    # Create a new 'group' column based on your rules

    def assign_group(row):
        if row['method'] == 'lora' and row['update_matrices'] == 'standard':
            return 'LoRA (Classic)'
        elif row['method'] == 'lora' and row['update_matrices'] == 'all_matrices':
            return 'LoRA (All)'
        elif row['method'] == 'ClassicalSurgery':
            return 'Full Finetuning'
        else:
            return None

    df_long['group'] = df_long.apply(assign_group, axis=1)
    # Filter out rows not in any of these groups.
    df_long = df_long[df_long['group'].notnull()]

    # Aggregate if multiple entries exist per combination
    df_plot = df_long.groupby(['dataset', 'start_layer', 'group', 'metric'])['value'].mean().reset_index()

    # Set up the plot: one subplot per dataset.
    datasets = [dataset for dataset in DATASETS.keys() if DOMAIN[dataset] == domain]
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets // 3, 3, figsize=(6 * 3, 6 * n_datasets // 3), sharey=False)

    metric_list = ['box', 'point', 'ais']
    hatch_dict = {
        'LoRA (Classic)': '///',   # hatched
        'LoRA (All)': 'oo',          # dotted
        'Full Finetuning': '',    # no pattern
    }

    # Fixed order for groups (if present)
    groups_order = ['LoRA (All)', 'LoRA (Classic)', 'Full Finetuning']

    # Plot each dataset in its own subplot
    for ax, dataset in zip(axes.flatten(), datasets):
        subset = df_plot[df_plot['dataset'] == dataset]
        # Treat start_layer as categorical by sorting and then assigning an index for even spacing.
        unique_layers = ['100%', '50%', '25%', '8%', 'Freeze Encoder (0%)']
        layer_positions = {layer: i for i, layer in enumerate(unique_layers)}

        n_groups = len(groups_order)
        cluster_width = 0.35  # total width of the cluster
        offsets = np.linspace(-cluster_width/2, cluster_width/2, n_groups)

        for layer in unique_layers:
            xpos_base = layer_positions[layer]
            for i, grp in enumerate(groups_order):
                grp_data = subset[(subset['start_layer'] == layer) & (subset['group'] == grp)]
                xpos = xpos_base + offsets[i]
                # Draw overlapping bars for each metric with the method's hatch pattern.
                metric_vals = {}
                for metric in metric_list:
                    row = grp_data[grp_data['metric'] == metric]
                    val = row['value'].values[0] if not row.empty else 0
                    metric_vals[metric] = val
                
                for metric, val in sorted(metric_vals.items(), key=lambda item: item[1], reverse=True):
                    ax.bar(xpos, val, width=0.18, color=CUSTOM_PALETTE[metric], alpha=1,
                        hatch=hatch_dict[grp], edgecolor='black')

        # Use evenly spaced ticks based on the number of unique layers and label them with the actual start_layer values.
        ax.set_xticks(list(layer_positions.values()))
        ax.set_xticklabels(unique_layers)
        ax.set_title(f'{DATASETS[dataset]}', fontweight='bold', fontsize=15)
        plt.setp(ax.get_xticklabels(), fontstyle='italic')
        ax.set_xlabel('Late Freezing Percentage')
        if DOMAIN[dataset] == "microscopy":
            ax.set_ylabel("Mean Segmentation Accuracy", fontsize=12, fontweight='bold')
            metric_names = {'ais': 'AIS', 'box': 'Box', 'point': 'Point'}
        else:
            ax.set_ylabel('Dice Similarity Coefficient', fontsize=12, fontweight='bold')
            metric_names = {'box': 'Box', 'point': 'Point'}
        
        # Add legend using one patch per metric
        metric_handles = [Patch(facecolor=CUSTOM_PALETTE[m], label=metric_names[m], alpha=0.7) for m in metric_names.keys()]

    handles = []
    labels = []
    for grp in groups_order:
        patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_dict[grp], label=grp)
        handles.append(patch)
        labels.append(grp)
    handles = handles + metric_handles
    metric_labels = [metric_handle.get_label() for metric_handle in metric_handles]
    labels = labels + metric_labels
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=8)

    plt.tight_layout()
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.99])  # Adjust space for the legend

    plt.savefig(f'../../results/figures/figure18_{domain}.png', dpi=300)
    plt.savefig(f'../../results/figures/figure18_{domain}.pdf')


def main():

    data = pd.read_csv("../../results/late_lora_all.csv")
    plot_late_lora(data, domain="medical")
    plot_late_lora(data, domain="microscopy")


if __name__ == '__main__':
    main()
