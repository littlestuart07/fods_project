import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot K-Means clusters as a PCA diagram")
    parser.add_argument(
        "--input",
        default="combined_cleaned_kmeans.csv",
        help="Input CSV path (must include a 'cluster' column)",
    )
    parser.add_argument("--output", default="kmeans_clusters_pca.png", help="Output image path")
    parser.add_argument("--sample", type=int, default=50000, help="Rows to plot (speed/clarity)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for sampling/PCA")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "cluster" not in df.columns:
        raise ValueError("Input file must contain a 'cluster' column. Use combined_cleaned_kmeans.csv.")

    # Sample for faster plotting and a cleaner diagram
    if args.sample and args.sample > 0 and len(df) > args.sample:
        df_plot = df.sample(n=args.sample, random_state=args.random_state)
    else:
        df_plot = df

    y = df_plot["cluster"].astype(int)
    X = df_plot.drop(columns=[c for c in ["cluster", "heart_disease"] if c in df_plot.columns])

    # Recreate the clustering feature matrix (one-hot + impute + scale) for PCA projection
    X = pd.get_dummies(X, drop_first=True)
    X_imputed = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    X_pca = PCA(n_components=2, random_state=args.random_state).fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=6, alpha=0.5, cmap="tab10")
    plt.title("K-Means clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved diagram: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

