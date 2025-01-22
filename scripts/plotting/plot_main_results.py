import pandas as pd
import matplotlib.pyplot as plt

DATASET_MAPPING = {
    "covid_if": "CovidIF",
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
    "ais": "#FF8B94",
    "point": "#9C89E2",
    "box": "#FFE278",
    "ip": "#56A4C4",
    "ib": "#82D37E",
}


def plot_results(df):
    metrics = ['ais', 'point', 'box', 'ip', 'ib'] 
    df['model'] = df['model'].replace({'vit_b_lm': r'$\mu$-SAM', 'vit_b_em_organelles': r'$\mu$-SAM'})
    df['model'] = df['model'].replace({'vit_b': 'SAM'})
    modality_order = list(MODALITY_MAPPING.keys())
    df = df.sort_values(
        by='modality', key=lambda x: x.map(lambda val: modality_order.index(val) if val in modality_order
                                           else len(modality_order))
    )
    #df = df.sort_values(
    #    by='dataset', key=lambda x: x.map(lambda val: dataset_order.index(val) if val in dataset_order
    #                                      else len(dataset_order))
    #)

    df['modality'] = df['modality'].replace(MODALITY_MAPPING)
    df['dataset'] = df['dataset'].replace(DATASET_MAPPING)

    df = df[df['dataset'] != 'LIVECell']

    # Unique datasets and models
    datasets = DATASET_MAPPING.values()
    models = df['model'].unique()
    # Define markers for models
    model_markers = {
        r"$\mu$-SAM": "^",
        "SAM": "x"
    }
    # Create a plot for each dataset
    fig, axes = plt.subplots(2, 3, figsize=(15, 12), sharex=False, sharey=True)

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
                    color=CUSTOM_PALETTE[metric]
                )

            # Highlight the top 3 PEFT methods for each metric
        for metric in metrics:
            # Find the top 3 methods
            circle_sizes = {0: 250, 1: 125, 2: 40}
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

    fig.tight_layout(rect=[0.05, 0.04, 0.97, 0.98])  # Adjust space for the legend
    # fig.suptitle("Comparison of PEFT methods", fontsize=16, y=0.98)
    fig.subplots_adjust(hspace=0.44)

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
        handles, labels, loc='lower center', ncol=10, fontsize=13, 
    )
    plt.text(x=-25.5, y=0.8, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=15)
    plt.savefig('../../results/figures/main_results.png', dpi=300)


if __name__ == "__main__":
    df = pd.read_csv('../../results/main_results_new.csv')
    plot_results(df)
