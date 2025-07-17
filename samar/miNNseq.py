import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances


def miNNseq(X: np.array, n_neighbors: int, squared: bool = True, lamda: float = 0.3):
    X = X.copy()
    imputed_indices = np.where(np.isnan(X))

    for i, j in zip(imputed_indices[0], imputed_indices[1]):
        distances = nan_euclidean_distances(
            X, X, squared=squared, missing_values=np.nan
        )

        neighbors_idx = np.argsort(distances[i])
        neighbors_idx = neighbors_idx[
            ~np.isin(
                neighbors_idx, imputed_indices[0][np.where(imputed_indices[1] == j)[0]]
            )
        ]
        neighbors_idx = neighbors_idx[:n_neighbors]
        weights = (1 / np.sqrt(2 * np.pi)) * np.exp(
            (-1 / 2) * np.power(distances[i][neighbors_idx] / lamda, 2)
        )
        weights = weights + 1 if np.all(weights == 0) else weights
        normalized_weights = weights / np.sum(weights)

        X[i, j] = np.average(X[neighbors_idx, j], weights=normalized_weights)

    return X


if __name__ == "__main__":
    data = np.array(
        [[1, 2, np.nan], [3, np.nan, 3], [7, 6, 5], [np.nan, 8, 7], [2, np.nan, 4]]
    )

    imputed_data = miNNseq(data, n_neighbors=3)
    print("Imputed Data:\n", imputed_data)
