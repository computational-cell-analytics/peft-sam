import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap

COLORS = ["#045275", "#7CCBA2", "#90477F", "#089099", "#F0746E"]
TASKS = {"ais": "AIS", "single point": "Point", "single box": "Box", "ip": r"$I_{\mathbfit{P}}$", "ib": r"$I_{\mathbfit{B}}$"}


def plot_results(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    all_handles = []
    all_labels = []

    df['alpha'] = df['alpha'].replace('learnable_scalar', 'learnable')
    for i, metric in enumerate(TASKS.keys()):

        ax = sns.lineplot(data=df, x="dropout", y=metric, hue="alpha", style="projection size",
                          markers=True, dashes=False, palette=COLORS, ax=axes.flatten()[i], markersize=10)
        handles, labels = ax.get_legend_handles_labels()
        if not all_handles and not all_labels:
            all_handles, all_labels = handles, labels
        ax.get_legend().remove()  # Remove legend from individual plots
        ax.set_title(f"{TASKS[metric]}", fontdict={"fontsize": 12, "fontweight": "bold"})
        ax.set_xlabel("Dropout")
        ax.set(ylabel=None)

    # Add a combined legend in the empty 6th subplot
    plt.subplot(2, 3, 6)
    plt.axis('off')
    fig.legend(all_handles, all_labels, loc="lower center", ncol=9)

    fig.tight_layout(rect=[0.04, 0.05, 1, 0.95])  # Adjust space for the legend
    fig.subplots_adjust(hspace=0.3)
    plt.text(x=-2.55, y=0.6, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold", fontsize=15)
    plt.savefig("../../results/figures/adaptformer_1.pdf", dpi=300)


def plot_heatmap(df):
    custom_palette = ['#000000', '#66C2C5', '#B0B0B0']
    cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_palette)

    mean_over_dropout = df.groupby(['alpha', 'projection size']).mean().reset_index()
    df['alpha'] = df['alpha'].replace('learnable_scalar', 'learnable')


    # Heatmap plotting function
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for j, task in enumerate(TASKS.keys()):
        ax = axes.flatten()[j]
        pivot_table = mean_over_dropout.pivot(index="projection size", columns="alpha", values=task)
        sns.heatmap(pivot_table, ax=ax, cmap=cmap, cbar_kws={'shrink': 0.75}, annot=True)
        ax.set_title(f"{TASKS[task]}", fontdict={"fontsize": 12, "fontweight": "bold"})
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Projection Size")
    empty_ax = axes.flatten()[-1]
    empty_ax.axis('off')
    plt.tight_layout()
    plt.savefig("../../results/figures/adaptformer_2.pdf", dpi=300)


def main():
    df = pd.read_csv("../../results/results_adaptformer.csv")
    plot_results(df)
    plot_heatmap(df)

if __name__ == "__main__":
    main()