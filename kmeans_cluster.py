import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> int:
    parser = argparse.ArgumentParser(description="Run K-Means clustering on combined_cleaned.csv")
    parser.add_argument("--input", default="combined_cleaned.csv", help="Input CSV path")
    parser.add_argument("--output", default="combined_cleaned_kmeans.csv", help="Output CSV path")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k to try (inclusive)")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum k to try (inclusive)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for KMeans")
    parser.add_argument(
        "--silhouette-sample",
        type=int,
        default=10000,
        help="Rows to sample for silhouette (speed). Use 0 for full data (slow).",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if "heart_disease" in data.columns:
        X = data.drop(columns=["heart_disease"])
    else:
        X = data.copy()

    X = pd.get_dummies(X, drop_first=True)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    k_values = list(range(args.k_min, args.k_max + 1))
    if len(k_values) < 1:
        raise ValueError("No k values to try. Check --k-min/--k-max.")

    sil_scores: list[float] = []
    inertias: list[float] = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=args.random_state, n_init="auto")
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        if args.silhouette_sample and args.silhouette_sample > 0:
            sample_size = min(int(args.silhouette_sample), X_scaled.shape[0])
            sil = silhouette_score(
                X_scaled, labels, sample_size=sample_size, random_state=args.random_state
            )
        else:
            sil = silhouette_score(X_scaled, labels)
        sil_scores.append(float(sil))

    best_k = int(k_values[int(np.argmax(sil_scores))])
    print(f"Tried k={k_values}")
    print(f"Silhouette: {[round(s, 4) for s in sil_scores]}")
    print(f"Inertia: {[round(i, 2) for i in inertias]}")
    print(f"Best k by silhouette: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=args.random_state, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    out = data.copy()
    out["cluster"] = clusters
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Cluster counts:")
    print(out["cluster"].value_counts().sort_index())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

