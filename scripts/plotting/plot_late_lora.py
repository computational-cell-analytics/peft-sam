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
    'psfhs': 'PSFHS',
    'hpa': 'HPA'
}

DOMAIN = {
    "hpa": "microscopy",
    "psfhs": "medical"
}


def extract_results(experiment_folder):

    result_folders = glob(os.path.join(experiment_folder, '**', 'results'), recursive=True)
    results = []

    for result_path in result_folders:

        method = result_path.split('/')[-5]
        update_matrices = result_path.split('/')[-4]
        start_layer = result_path.split('/')[-3]
        dataset = result_path.split('/')[-2]

        # Load results
        iterative_prompting_box = os.path.join(
            result_path, 'iterative_prompting_without_mask', 'iterative_prompts_start_box.csv'
        )
        iterative_prompting_point = os.path.join(
            result_path, 'iterative_prompting_without_mask', 'iterative_prompts_start_point.csv'
        )
        instance_segmentation_file = os.path.join(result_path, 'instance_segmentation_with_decoder.csv')

        ais, single_box, single_point, ib, ip = None, None, None, None, None

        if os.path.exists(iterative_prompting_box):
            box_df = pd.read_csv(iterative_prompting_box)
            single_box = box_df["mSA"].iloc[0] if not box_df.empty else None
            ib = box_df["mSA"].iloc[-1] if not box_df.empty else None

        if os.path.exists(iterative_prompting_point):
            point_df = pd.read_csv(iterative_prompting_point)
            single_point = point_df["mSA"].iloc[0] if not point_df.empty else None
            ip = point_df["mSA"].iloc[-1] if not point_df.empty else None

        if os.path.exists(instance_segmentation_file):
            instance_df = pd.read_csv(instance_segmentation_file)
            ais = instance_df["mSA"].iloc[0] if "mSA" in instance_df.columns else None

        results.append({
            "dataset": dataset,
            "method": method,
            "update_matrices": update_matrices,
            "start_layer": int(start_layer.split('_')[-1]) + 1,
            "box": single_box,
            "point": single_point,
            "ib": ib,
            "ip": ip,
            "ais": ais
        })

    return pd.DataFrame(results)


def plot_late_lora(data):
    # metrics = ['box', 'point', 'ib', 'ip', 'ais']
    metrics = ['box', 'ais', 'point']
    df_long = pd.melt(
        data,
        id_vars=["dataset", "method", "update_matrices", "start_layer"],
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    )

    replace_layer = {'1': '100%', '7': '50%', '10': '25%', '12': '8%', 'freeze encoder': 'Freeze Encoder (0%)'}
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
    datasets = df_plot['dataset'].unique()
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6), sharey=False)

    if n_datasets == 1:
        axes = [axes]

    metric_list = ['box', 'ais', 'point']
    hatch_dict = {
        'LoRA (Classic)': '///',   # hatched
        'LoRA (All)': 'oo',          # dotted
        'Full Finetuning': ''    # no pattern
    }

    # Fixed order for groups (if present)
    groups_order = ['LoRA (All)', 'LoRA (Classic)', 'Full Finetuning']

    # Plot each dataset in its own subplot
    for ax, dataset in zip(axes, datasets):
        subset = df_plot[df_plot['dataset'] == dataset]
        # Treat start_layer as categorical by sorting and then assigning an index for even spacing.
        unique_layers = ['100%', '50%', '25%', '8%', 'Freeze Encoder (0%)']
        layer_positions = {layer: i for i, layer in enumerate(unique_layers)}

        n_groups = len(groups_order)
        cluster_width = 0.36  # total width of the cluster
        offsets = np.linspace(-cluster_width/2, cluster_width/2, n_groups)

        for layer in unique_layers:
            xpos_base = layer_positions[layer]
            for i, grp in enumerate(groups_order):
                grp_data = subset[(subset['start_layer'] == layer) & (subset['group'] == grp)]
                xpos = xpos_base + offsets[i]
                # Draw overlapping bars for each metric with the method's hatch pattern.
                for metric in metric_list:
                    row = grp_data[grp_data['metric'] == metric]
                    val = row['value'].values[0] if not row.empty else 0
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
        else:
            ax.set_ylabel('Dice Similarity Coefficient', fontsize=12, fontweight='bold')

        metric_names = {'ais': 'AIS', 'box': 'Box', 'point': 'Point'}
        # Add legend using one patch per metric
        metric_list = ['ais', 'box', 'point']
        metric_handles = [Patch(facecolor=CUSTOM_PALETTE[m], label=m, alpha=0.7) for m in metric_list]
        # metric_names = ['AIS', 'Point', 'Box', r'$I_{\mathbfit{P}}$', r'$I_{\mathbfit{B}}$']
        metric_names = ['AIS', 'Box', 'Point']

    handles = []
    labels = []
    for grp in groups_order:
        patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_dict[grp], label=grp)
        handles.append(patch)
        labels.append(grp)
    handles = handles + metric_handles
    labels = labels + metric_names
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=8)

    plt.tight_layout()
    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.99])  # Adjust space for the legend

    plt.savefig('../../results/figures/late_lora_results.svg', dpi=300)


def main():

    #experiment_folder = '/scratch/usr/nimcarot/sam/experiments/peft'
    #data = extract_results(experiment_folder)
    #data.to_csv("../../results/late_lora_results.csv", index=False)

    data = pd.read_csv("../../results/late_lora_results.csv")
    plot_late_lora(data)


if __name__ == '__main__':
    main()
