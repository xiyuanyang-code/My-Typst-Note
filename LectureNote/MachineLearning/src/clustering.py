import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.datasets import make_blobs


class ClusterModel:
    """
    An abstract/factory class for initializing, training, and predicting
    with different clustering models (GMM or KMeans++).

    This module is designed to be independent of data loading and evaluation logic.
    """

    def __init__(
        self, n_clusters: int, model_type: str = "GMM", random_state: int = 42
    ):
        """
        Initializes the clustering model instance.

        Args:
            n_clusters (int): The expected number of clusters (K).
            model_type (str): The type of model to use ('GMM' or 'KMeans').
            random_state (int): The random seed for reproducibility.

        Raises:
            ValueError: If an unsupported model_type is provided.
        """
        self.n_clusters = n_clusters
        self.model_type = model_type
        self.model = self._initialize_model(random_state)

    def _initialize_model(self, random_state):
        """
        Internal method to create and return the scikit-learn model instance.
        ! now only supports GMM and KMeans for clustering
        """
        if self.model_type == "GMM":
            return GaussianMixture(
                n_components=self.n_clusters, random_state=random_state
            )
        elif self.model_type == "KMeans":
            return KMeans(
                n_clusters=self.n_clusters,
                # use k-means++ initialization
                init="k-means++",
                random_state=random_state,
                n_init="auto",
            )
        else:
            raise ValueError("Unsupported model type. Please choose 'GMM' or 'KMeans'.")

    def fit(self, X: np.ndarray):
        """
        Trains the underlying clustering model.

        Args:
            X (np.ndarray): The training data features.
        """
        print(f"--- Training {self.model_type} model (K={self.n_clusters}) ---")
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for the given data points.

        Args:
            X (np.ndarray): The data to predict on.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        return self.model.predict(X)

    def get_model(self):
        """
        Returns the underlying scikit-learn model instance for inspection.
        """
        return self.model


class DataProcessor:
    """
    Utility class for loading, splitting, and preprocessing data,
    specifically handling feature scaling for clustering models.

    It is decoupled from the model training and prediction logic.
    """

    def __init__(self):
        """
        Initializes the data processor, including the standard scaler.
        """
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_test_true = None

    def load_and_preprocess_data(
        self, X: np.ndarray, y: np.ndarray = None, test_size: float = 0.3
    ):
        """
        Splits data into training/testing sets and performs feature scaling.

        Args:
            X (np.ndarray): The raw feature data.
            y (np.ndarray, optional): The true labels (if available, used for external evaluation).
            test_size (float): The proportion of data to use for the test set.

        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        # Split data; includes true labels if provided for later evaluation
        if y is not None:
            X_train, X_test, _, self.y_test_true = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)

        # Fit scaler on training data and transform both sets
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled

        print(f"Train set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Retrieves the processed test data and true labels.

        Returns:
            tuple: (X_test, y_test_true)
        """
        return self.X_test, self.y_test_true

class Prediction:
    """
    The main driver responsible for coordinating data, models, execution, and evaluation.

    It remains agnostic to the specific model implementation or data scaling method.
    """

    def __init__(self, processor: DataProcessor):
        """
        Initializes the framework with a DataProcessor instance.

        Args:
            processor (DataProcessor): The data processing module instance.
        """
        self.processor = processor
        self.models = {}

    def train_and_predict(
        self, model_name: str, model_type: str, n_clusters: int
    ) -> np.ndarray:
        """
        Instantiates, trains a model, and returns test set predictions.

        Args:
            model_name (str): A unique name for the model instance.
            model_type (str): The type of model ('GMM' or 'KMeans').
            n_clusters (int): The number of clusters (K).

        Returns:
            np.ndarray: The predicted labels for the test set.
        """
        X_train, X_test = self.processor.X_train, self.processor.X_test

        if X_train is None:
            raise RuntimeError(
                "Data not loaded or preprocessed. Please run load_and_preprocess_data first."
            )

        # Instantiate and train the model
        model_instance = ClusterModel(n_clusters, model_type)
        model_instance.fit(X_train)

        self.models[model_name] = model_instance

        # Predict on the test set
        y_pred = model_instance.predict(X_test)
        print(f"--- {model_name} prediction complete ---")
        return y_pred

    def evaluate(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray = None,
    ):
        """
        Evaluates the clustering results using internal and external metrics.

        Args:
            model_name (str): The name of the model being evaluated.
            X_test (np.ndarray): The feature data of the test set.
            y_pred (np.ndarray): The predicted cluster labels.
            y_true (np.ndarray, optional): The true labels (if available).
        """
        print(f"\n======== {model_name} Evaluation Results ========")

        # 1. Internal Metric: Silhouette Score
        try:
            score = silhouette_score(X_test, y_pred)
            print(f"Silhouette Score: {score:.4f} (Internal Consistency)")
        except Exception as e:
            print(f"Could not calculate Silhouette Score: {e}")

        # 2. External Metric: Homogeneity Score (Requires true labels)
        if y_true is not None:
            score = homogeneity_score(y_true, y_pred)
            print(f"Homogeneity Score: {score:.4f} (External Metric)")
        else:
            print("True labels not provided, skipping external metrics.")


def run_clustering_framework():
    """
    Runs the complete clustering prediction framework using GMM and KMeans++.
    """
    K_TRUE = 4
    X, y = make_blobs(n_samples=500, centers=K_TRUE, cluster_std=1.0, random_state=42)

    # 2. Initialize and run the Data Processor
    data_processor = DataProcessor()
    data_processor.load_and_preprocess_data(X, y=y, test_size=0.3)

    X_test, y_true = data_processor.get_test_data()

    # 3. Initialize the Prediction Framework
    framework = Prediction(data_processor)
    K = K_TRUE

    # --- Run GMM Model ---
    GMM_NAME = "GMM_Model"
    y_pred_gmm = framework.train_and_predict(GMM_NAME, "GMM", K)
    framework.evaluate(GMM_NAME, X_test, y_pred_gmm, y_true)

    # --- Run K-Means++ Model ---
    KMEANS_NAME = "KMeansPP_Model"
    y_pred_kmeans = framework.train_and_predict(KMEANS_NAME, "KMeans", K)
    framework.evaluate(KMEANS_NAME, X_test, y_pred_kmeans, y_true)


if __name__ == "__main__":
    run_clustering_framework()
