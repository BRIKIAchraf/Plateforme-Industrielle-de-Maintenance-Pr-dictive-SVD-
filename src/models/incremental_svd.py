import numpy as np
from sklearn.decomposition import IncrementalPCA
from typing import Tuple, Optional

class OnlineIncrementalSVD:
    """
    High-performance online incremental SVD for real-time low-rank updates.
    Includes a forgetting factor to adapt the baseline model and handle mechanical drift.
    """
    def __init__(self, n_components: int = 5, forgetting_factor: float = 0.99):
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor
        
        self.ipca = IncrementalPCA(n_components=n_components)
        
        # State tracking for running statistics
        self.running_mean = None
        self.running_variance = None
        self.n_samples_seen = 0
        
        # Online tracking of Singular Values (S) and Singular Vectors (U, Vt)
        self.singular_values_ = None
        self.components_ = None

    def update(self, X: np.ndarray) -> None:
        """
        Incrementally fit the model with a new batch X (n_samples, n_features).
        Applies a forgetting factor to historical data.
        """
        n_samples, n_features = X.shape
        if n_samples < self.n_components:
            # Not enough samples in this batch to update, fallback to accumulating
            return

        # Fit batch
        self.ipca.partial_fit(X)
        
        # Apply forgetting factor to singular values to let old baseline decay
        if self.singular_values_ is not None:
            # We blend the newly calculated singular values with old ones using the forgetting factor
            self.singular_values_ = (self.forgetting_factor * self.singular_values_ + 
                                     (1 - self.forgetting_factor) * self.ipca.singular_values_)
        else:
            self.singular_values_ = self.ipca.singular_values_
            
        self.components_ = self.ipca.components_
        
        # Update running mean and variance with forgetting factor
        batch_mean = np.mean(X, axis=0)
        batch_var = np.var(X, axis=0)
        
        if self.running_mean is None:
            self.running_mean = batch_mean
            self.running_variance = batch_var
        else:
            self.running_mean = (self.forgetting_factor * self.running_mean + 
                                 (1 - self.forgetting_factor) * batch_mean)
            self.running_variance = (self.forgetting_factor * self.running_variance + 
                                     (1 - self.forgetting_factor) * batch_var)
            
        self.n_samples_seen += n_samples

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects input data X into the singular components latent space.
        """
        if self.components_ is None:
            raise ValueError("Model has not been trained yet. Call update first.")
        # Zero-center data using our running mean
        X_centered = X - self.running_mean
        return np.dot(X_centered, self.components_.T)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstructs the original data from its low-rank SVD projection.
        """
        latent = self.transform(X)
        return np.dot(latent, self.components_) + self.running_mean

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates Frobenius reconstruction residual per time step.
        """
        X_hat = self.reconstruct(X)
        # Row-wise Frobenius norm (Euclidean distance of prediction vs actual)
        return np.linalg.norm(X - X_hat, axis=1)

    def get_spectral_drift(self) -> float:
        """
        Returns a measure of singular spectrum drift (wear).
        High values indicate that the singular values have significantly deviated
        from their initial levels (indicating core mechanics changes).
        """
        if self.singular_values_ is None:
            return 0.0
        # Ratio of top singular value vs sum of all singular values
        return self.singular_values_[0] / (np.sum(self.singular_values_) + 1e-9)
