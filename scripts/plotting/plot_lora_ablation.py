import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

COLORS = ["#045275", "#7CCBA2", "#90477F", "#089099", "#F0746E"]

TASKS = {"ais": "AIS", "single point": "Point", "single box": "Box", "ip": r"$I_{\mathbfit{P}}$", "ib": r"$I_{\mathbfit{B}}$"}
MARKERS = ["o", "^", "X"]
subtitle_font = {"fontsize": 12, "fontweight": "bold"}
suptitle_font = {"fontsize": 16, "fontweight": "bold"}
label_font = {"fontsize": 12}
legend_font = {"fontsize": 10}


def plot_lora_a(df):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False, sharey=False)
    for i, task in enumerate(TASKS.keys()):
        ax = axes.flatten()[i]
        for alpha, color in zip(df['alpha'].unique(), COLORS[:3]):
            subset = df[df['alpha'] == alpha]
            ax.plot(subset['rank'], subset[task], marker='o', label=alpha, color=color)

        ax.set_title(TASKS[task], fontdict=subtitle_font)
        ax.set_xlabel("Rank", fontsize=label_font['fontsize'])
        # ax.set_ylabel("Segmentation Accuracy", fontsize=label_font['fontsize'])
        # ax.legend(title=r"$\alpha$", fontsize=legend_font['fontsize'])

    # Remove grid and axis for the last subplot, add legend
    empty_ax = axes.flatten()[-1]
    empty_ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, title=r"$\alpha$", fontsize=10, frameon=True)

    # Adjust overall layout
    fig.tight_layout(rect=[0.05, 0.05, 0.99, 0.99])  # Adjust space for the legend
    fig.subplots_adjust(hspace=0.33)
    plt.text(x=-1.3, y=1.0, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=16)
    plt.savefig("../../results/figures/lora_1.pdf", dpi=300)


def plot_lora_b(df):
    custom_palette = ['#000000', '#66C2C5', '#B0B0B0']
    cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_palette)

    # Heatmap plotting function
    fig, axes = plt.subplots(5, 2, figsize=(10, 17))
    for j, task in enumerate(TASKS.keys()):
        for i, rank in enumerate([1, 32]):
            ax = axes[j, i]
            pivot_table = df[df['rank'] == rank].pivot(index="alpha", columns="lr", values=task)
            sns.heatmap(pivot_table, ax=ax, cmap=cmap, cbar_kws={'shrink': 0.75}, annot=True)
            ax.set_title(f"{TASKS[task]} (Rank={rank})", fontsize=10)
            ax.set_ylabel(r"$\alpha$")
            ax.set_xlabel("Learning Rate")

    plt.tight_layout()
    plt.savefig("../../results/figures/lora_2.pdf", dpi=300)


def plot_lora_c(df):
    # Replace model names with more readable labels
    df['base model'] = df['base model'].replace({'vit_b_lm': r'$\mu$SAM', 'vit_b_em_organelles': r'$\mu$SAM'})
    df['base model'] = df['base model'].replace({'vit_b': 'SAM'})

    models = df['base model'].unique()

    # Define markers for models
    model_markers = {
        r"$\mu$SAM": "^",
        "SAM": "x"
    }
    dataset_mapping = {
        "covid_if": "Covid-IF",
        "mitolab_glycolytic_muscle": "MitoLab",
        "platy_cilia": "Platynereis",
        "orgasegment": "OrgaSegment",
    }

    df['dataset'] = df['dataset'].replace(dataset_mapping)
    datasets = dataset_mapping.values()
    # Create a plot for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=False)

    for row, dataset in enumerate(datasets):
        ax = axes.flatten()[row]
        for model in models:
            # Filter data for the current model
            model_data = df[df['base model'] == model]
            # Plot each metric with the custom color palette and model markers
            for i, task in enumerate(TASKS.keys()):
                dataset_data = model_data[model_data['dataset'] == dataset]
                ax.plot(
                    dataset_data['alpha'],
                    dataset_data[task],
                    marker=model_markers[model],
                    label=f"{dataset} ({model})",
                    color=COLORS[i]
                )
        for i, task in enumerate(TASKS.keys()):
            # Find the top alpha
            top_indices = dataset_data.nlargest(1, task).index
            point_x = dataset_data.loc[top_indices[0], 'alpha']
            point_y = dataset_data.loc[top_indices[0], task]
            ax.scatter([point_x], [point_y], s=150, color=COLORS[i], alpha=0.3)


        # Set titles and labels
        ax.set_title(dataset, fontsize=12)  # Remove underscores, capitalize
        ax.tick_params(axis='x')
        ax.set_xticks(model_data['alpha'])
        ax.set_xticklabels(model_data['alpha'])

    # Adjust overall layout
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust space for the legend
    fig.subplots_adjust(hspace=0.2)
    plt.text(x=-11, y=0.65, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=13)

    # Create a customized legend
    # Metric legend (colors only, no markers)
    metric_handles = [
        plt.Line2D([0], [0], color=COLORS[i], lw=2) for i in range(len(TASKS.keys()))
    ]
    metric_labels = list(TASKS.values())

    # Model legend (markers in black)
    model_handles = [
        plt.Line2D([0], [0], color="black", marker=model_markers[model], linestyle='') 
        for model in models
    ]
    model_labels = models

    # Ranking legend (transparent circles for ranks 1, 2, and 3)
    ranking_handles = [
        plt.scatter([], [], s=150, color="gray", alpha=0.3, label="Best Accuracy"),
    ]
    ranking_labels = ["Best Accuracy"]

    # Combine legends
    handles = metric_handles + model_handles + ranking_handles
    labels = metric_labels + list(model_labels) + ranking_labels

    # Add the legend to the figure
    fig.legend(
        handles, labels, loc='lower center', ncol=10, fontsize=10,
    )
    plt.savefig('../../results/figures/lora_3.pdf', dpi=300)


def main():
    df_a = pd.read_csv("../../results/lora/results_a.csv")
    plot_lora_a(df_a)
    df_b = pd.read_csv("../../results/lora/results_b.csv")
    plot_lora_b(df_b)
    df_c = pd.read_csv("../../results/lora/results_d.csv")
    plot_lora_c(df_c)


if __name__ == "__main__":
    main()
