import pandas as pd
import matplotlib.pyplot as plt

DATASET_MAPPING = {
    "covid_if": "CovidIF",
    "mitolab_glycolytic_muscle": "MitoLab",
    "platy_cilia": "Platynereis",
    "orgasegment": "OrgaSegment",
    "gonuclear": "GoNuclear",
    "hpa": "HPA",
    "livecell": "LIVECell",
}
MODALITY_MAPPING = {
    "vanilla": "Base Model",
    "generalist": "Base Model",
    "full_ft": "Full Finetuning",
    "lora": "LoRA",
    "AttentionSurgery": "Attention Tuning",
    "BiasSurgery": "Bias Tuning",
    "LayerNormSurgery": "Layernorm Tuning",
    "fact": "FacT",
    "adaptformer": "AdaptFormer",
    "freeze_encoder": "Freeze Encoder",
    "ssf": "SSF"
}
CUSTOM_PALETTE = {
    "ais": "#FF8B94",
    "ip": "#56A4C4",
    "ib": "#82D37E",
    "single box": "#FFE278",
    "single point": "#9C89E2"
}


def plot_results(df):
    metrics = ['ais', 'ip', 'ib', 'single box', 'single point'] 
    df['model'] = df['model'].replace({'vit_b_lm': r'$\mu$-SAM', 'vit_b_em_organelles': r'$\mu$-SAM'})
    df['model'] = df['model'].replace({'vit_b': 'SAM'})
    custom_order = list(MODALITY_MAPPING.keys())
    df = df.sort_values(by='modality', key=lambda x: x.map(lambda val: custom_order.index(val) if val in custom_order else len(custom_order)))

    df['modality'] = df['modality'].replace(MODALITY_MAPPING)
    df['dataset'] = df['dataset'].replace(DATASET_MAPPING)

    # Unique datasets and models
    datasets = DATASET_MAPPING.values()
    models = df['model'].unique()
    # Define markers for models
    model_markers = {
        r"$\mu$-SAM": "*",
        "SAM": "x"
    }
    # Create a plot for each dataset
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=False, sharey=True)

    axes[3, 1].axis('off')  # Turn off the bottom-right subpl
    for row, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]

        ax = axes.flatten()[row]
        if dataset == 'livecell':
            models = ['SAM']
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
                    color=CUSTOM_PALETTE[metric]
                )

            # Highlight the top 3 PEFT methods for each metric
        for metric in metrics:
            # Find the top 3 methods
            top_indices = dataset_data.nlargest(3, metric).index
            for rank, idx in enumerate(top_indices):
                point_x = dataset_data.loc[idx, 'modality']
                point_y = dataset_data.loc[idx, metric]
                circle_size = 150 - rank * 50  # Size decreases with rank

                # Add a circle around the point
                ax.scatter(
                    [point_x], [point_y],
                    s=circle_size, color=CUSTOM_PALETTE[metric], alpha=0.5, linewidth=2
                )

        # Set titles and labels
        ax.set_title(dataset, fontsize=12)  # Remove underscores, capitalize
        ax.set_ylabel("Segmentation Accuracy", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticks(model_data['modality'])
        ax.set_xticklabels(model_data['modality'], rotation=45)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust space for the legend
    fig.suptitle("Comparison of PEFT methods", fontsize=16, y=0.98)
    fig.subplots_adjust(hspace=0.6)

    metric_handles = [
        plt.Line2D([0], [0], color=CUSTOM_PALETTE[metric], lw=2) for metric in metrics
    ]
    model_handles = [
        plt.Line2D([0], [0], color="black", marker=model_markers[model], linestyle='')
        for model in models
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
    labels = metrics + list(models) + ranking_labels

    # Add the legend to the figure
    fig.legend(
        handles, labels, loc='lower center', ncol=10, fontsize=10,
    )
    plt.savefig('../../results/figures/main_results.png', dpi=300)


if __name__ == "__main__":
    df = pd.read_csv('../../results/main_results.csv')
    plot_results(df)
