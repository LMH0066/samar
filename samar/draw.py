import matplotlib.pyplot as plt
import seaborn as sns


def lineplot(
    data,
    x=None,
    y=None,
    hue=None,
    xticklabels=None,
    output_path: str = None,
    figsize: tuple = (24, 12),
):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.lineplot(data=data, x=x, y=y, hue=hue)
    ax = sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color="red")

    plt.legend(loc="lower right")
    plt.grid(False)

    if x:
        plt.xlabel(x)
    if y:
        plt.ylabel(y)
    if xticklabels:
        ax.set_xticklabels(labels=xticklabels)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()
