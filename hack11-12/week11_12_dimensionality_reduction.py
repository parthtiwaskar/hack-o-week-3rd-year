"""Week 11–12: visualize high-dimensional data with PCA and t-SNE.

Run with: python week11_12_dimensionality_reduction.py
Add --save-plot to save week11_12_embeddings.png instead of opening a window.
Requires numpy, matplotlib, and scikit-learn.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def make_high_dimensional_data() -> tuple[np.ndarray, np.ndarray]:
    """Make labeled data with four groups across twelve measurements."""
    features, labels = make_blobs(
        n_samples=480,
        centers=4,
        n_features=12,
        cluster_std=3.0,
        random_state=41,
    )
    return features, labels


def create_embeddings(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standardize features and create two different 2D projections."""
    scaled_features = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(scaled_features)

    tsne_embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=41,
    ).fit_transform(scaled_features)

    explained_variance = float(pca.explained_variance_ratio_.sum())
    return pca_embedding, tsne_embedding, explained_variance


def print_intuition(explained_variance: float) -> None:
    """Summarize what the two methods preserve and when to use them."""
    print("PCA: a linear projection onto directions with the most variance.")
    print(f"The first two PCA components capture {explained_variance:.1%} of total variance.")
    print("Use PCA for compact features, denoising, or a fast broad view of the data.")
    print("\nt-SNE: rearranges points so nearby high-dimensional neighbors tend to stay nearby.")
    print("Use t-SNE mainly to explore local clusters in a 2D/3D visualization.")
    print("Interpret t-SNE cautiously: axis values and distances between separated groups are not meaningful.")
    print("t-SNE is stochastic and is generally a visualization, not a reusable feature transform.")


def plot_embeddings(
    pca_embedding: np.ndarray,
    tsne_embedding: np.ndarray,
    labels: np.ndarray,
    output: Path | None = None,
) -> None:
    """Plot both embeddings with the same class colors for comparison."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plots = (
        (axes[0], pca_embedding, "PCA: global variance directions"),
        (axes[1], tsne_embedding, "t-SNE: local neighborhood view"),
    )

    for axis, embedding, title in plots:
        points = axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="viridis",
            s=22,
            alpha=0.8,
        )
        axis.set(title=title, xlabel="Component 1", ylabel="Component 2")
        axis.set_xticks([])
        axis.set_yticks([])

    figure.colorbar(points, ax=axes, label="Group", ticks=np.unique(labels))
    figure.suptitle("Two views of the same 12-dimensional dataset")
    figure.tight_layout()
    if output:
        figure.savefig(output, dpi=160, bbox_inches="tight")
        print(f"\nSaved comparison plot to {output}")
    else:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="save the comparison plot instead of opening a window",
    )
    args = parser.parse_args()

    features, labels = make_high_dimensional_data()
    pca_embedding, tsne_embedding, explained_variance = create_embeddings(features)
    print(f"Input shape: {features.shape}; each method maps rows to two coordinates.")
    print_intuition(explained_variance)

    assert pca_embedding.shape == tsne_embedding.shape == (len(labels), 2)
    assert 0.0 < explained_variance <= 1.0

    output = Path("week11_12_embeddings.png") if args.save_plot else None
    plot_embeddings(pca_embedding, tsne_embedding, labels, output)


if __name__ == "__main__":
    main()