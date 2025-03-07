import pandas as pd
import matplotlib.pyplot as plt

MEDICO_DATASET_MAPPING = {
    "amd_sd": "AMD-SD",
    "jsrt": "JSRT",
    "mice_tumseg": "Mice TumSeg",
    "papila": "Papila",
    "motum": "MOTUM",
    "psfhs": "PSFHS",
}

MICROSCOPY_DATASET_MAPPING = {
    "covid_if": "Covid-IF",
    "orgasegment": "OrgaSegment",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "mitolab_glycolytic_muscle": "MitoLab",
    "platy_cilia": "Platynereis",
}

MODALITY_MAPPING = {
    "vanilla": "Base Model",
    "generalist": "Base Model",
    "freeze_encoder": "Freeze Encoder",
    "LayerNormSurgery": "LN Tune",
    "BiasSurgery": "Bias Tune",
    "ssf": "SSF",
    "fact": "FacT",
    "qlora": "QLoRA",
    "lora": "LoRA",
    "adaptformer": "AdaptFormer",
    "AttentionSurgery": "Attn Tune",
    "full_ft": "Full Ft",
}

CUSTOM_PALETTE = {
    "ais": "#045275",
    "point": "#7CCBA2",
    "box": "#90477F",
    "ip": "#089099",
    "ib": "#F0746E",
}


def plot_results(df, domain):
    df['model'] = df['model'].replace({'vit_b': 'SAM'})
    modality_order = list(MODALITY_MAPPING.keys())
    df = df.sort_values(
        by='modality', key=lambda x: x.map(lambda val: modality_order.index(val) if val in modality_order
                                           else len(modality_order))
    )
    df['modality'] = df['modality'].replace(MODALITY_MAPPING)
    if domain == "microscopy":
        # metrics = ['ais', 'point', 'box', 'ip', 'ib'] 
        metrics = ['ais', 'point', 'box']
        # metric_names = ['AIS', 'Point', 'Box', r'$I_{\mathbfit{P}}$', r'$I_{\mathbfit{B}}$']
        metric_names = ['AIS', 'Point', 'Box']
        df['model'] = df['model'].replace({'vit_b_lm': r'$\mu$SAM', 'vit_b_em_organelles': r'$\mu$SAM'})
        df['dataset'] = df['dataset'].replace(MICROSCOPY_DATASET_MAPPING)
        datasets = MICROSCOPY_DATASET_MAPPING.values()
        models = ["$\mu$SAM", "SAM"]
        model_markers = {
            "SAM": "x",
            r"$\mu$SAM": "^",
            "MedicoSAM": "d"
        }
    elif domain == "medical":
        # metrics = ['point', 'box', 'ip', 'ib'] 
        metrics = ['point', 'box']
        # metric_names = ['Point', 'Box', r'$I_{\mathbfit{P}}$', r'$I_{\mathbfit{B}}$']
        metric_names = ['Point', 'Box']
        df['model'] = df['model'].replace({'vit_b_medical_imaging': 'MedicoSAM'})
        df['dataset'] = df['dataset'].replace(MEDICO_DATASET_MAPPING)
        datasets = MEDICO_DATASET_MAPPING.values()
        model_markers = {
            "MedicoSAM": "d",
            "SAM": "x"
        }

    models = df['model'].unique()
    # Create a plot for each dataset
    fig, axes = plt.subplots(3, 2, figsize=(10, 16), sharex=False, sharey=True)

    for row, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]
        ax = axes.flatten()[row]
        for model in models:
            # Filter data for the current model
            model_data = dataset_data[dataset_data['model'] == model]

            # Plot each metric with the custom color palette and model markers
            for metric in metrics:
                ax.plot(
                    model_data['modality'],
                    model_data[metric],
                    marker=model_markers[model],
                    label=f"{metric} ({model})",
                    color=CUSTOM_PALETTE[metric],
                    markersize=8
                )

            # Highlight the top 3 PEFT methods for each metric
        for metric in metrics:
            # Find the top 3 methods
            circle_sizes = {0: 450, 1: 300, 2: 200}
            top_indices = dataset_data.nlargest(3, metric).index
            for rank, idx in enumerate(top_indices):
                point_x = dataset_data.loc[idx, 'modality']
                point_y = dataset_data.loc[idx, metric]
                circle_size = circle_sizes[rank]  # Size decreases with rank

                # Add a circle airound the point
                ax.scatter(
                    [point_x], [point_y],
                    s=circle_size, color=CUSTOM_PALETTE[metric], alpha=0.5, linewidth=2
                )

        # Set titles and labels
        ax.set_title(dataset, fontsize=15)  # Remove underscores, capitalize
        ax.tick_params(axis='x', rotation=90)
        ax.set_xticks(model_data['modality'])
        ax.set_xticklabels(model_data['modality'], rotation=90, fontsize=13)
        ax.tick_params(axis='y', labelsize=11)

    fig.tight_layout(rect=[0.05, 0.04, 0.97, 0.98])  # Adjust space for the legend
    # fig.suptitle("Comparison of PEFT methods", fontsize=16, y=0.98)
    fig.subplots_adjust(hspace=0.52)

    metric_handles = [
        plt.Line2D([0], [0], color=CUSTOM_PALETTE[metric], lw=2) for metric in metrics
    ]
    model_handles = [
        plt.Line2D([0], [0], color="black", marker=model_markers[model], linestyle='')
        for model in model_markers.keys()
    ]
    # Ranking legend (transparent circles for ranks 1, 2, and 3)
    ranking_handles = [
        plt.scatter([], [], s=150, color="gray", alpha=0.3, label="1st Place"),
        plt.scatter([], [], s=100, color="gray", alpha=0.3, label="2nd Place"),
        plt.scatter([], [], s=50, color="gray", alpha=0.3, label="3rd Place")
    ]
    ranking_labels = ["1st Place", "2nd Place", "3rd Place"]

    # Combine legends
    handles = metric_handles + model_handles + ranking_handles
    labels = metric_names + list(model_markers.keys()) + ranking_labels

    # Add the legend to the figure
    # fig.legend(
    #     handles, labels, loc='lower center', ncol=10, fontsize=13,
    # )

    if domain == "microscopy":
        plt.text(x=-13.8, y=1.2, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=18)
    elif domain == "medical":
        plt.text(x=-13.8, y=1.4, s="Dice Similarity Coefficient", rotation=90, fontweight="bold", fontsize=18)
    plt.savefig(f'../../results/figures/results_{domain}_v2.svg')
    plt.savefig(f'../../results/figures/results_{domain}_v2.png', dpi=300)
    legend_fig = plt.figure()
    legend_ax = legend_fig.add_axes([0, 0, 1, 1])
    legend_ax.legend(handles, labels, ncol=10, fontsize=13)
    legend_ax.axis('off')
    legend_fig.savefig('../../results/figures/legend.svg', bbox_inches='tight')


if __name__ == "__main__":
    df_microscopy = pd.read_csv('../../results/main_results.csv')
    plot_results(df_microscopy, "microscopy")
    df_medical = pd.read_csv('../../results/medico_sam_v2.csv')
    plot_results(df_medical, "medical")
