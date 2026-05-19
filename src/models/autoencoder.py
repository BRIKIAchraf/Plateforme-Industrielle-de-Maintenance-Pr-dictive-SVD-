import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List

class NeuralAutoencoder(nn.Module):
    """
    Non-linear PyTorch Autoencoder for extracting robust latent representations
    and computing deep reconstruction-based anomaly scores.
    """
    def __init__(self, input_dim: int, latent_dim: int = 3, hidden_dims: List[int] = [8, 6]):
        super(NeuralAutoencoder, self).__init__()
        
        # Build Encoder
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = h_dim
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build Decoder
        decoder_layers = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            last_dim = h_dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class AutoencoderEngine:
    """
    Wrapper around the NeuralAutoencoder PyTorch module to handle training,
    inference, and anomaly score evaluation.
    """
    def __init__(self, input_dim: int, latent_dim: int = 3, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, X: np.ndarray, epochs: int = 5, batch_size: int = 32) -> float:
        """
        Train the autoencoder on a batch of sensor readings.
        """
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset_size = len(X)
        
        epoch_losses = []
        for epoch in range(epochs):
            permutation = torch.randperm(X_tensor.size()[0])
            epoch_loss = 0.0
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = X_tensor[indices]
                
                self.optimizer.zero_grad()
                reconstructed, _ = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
            
            epoch_losses.append(epoch_loss / dataset_size)
            
        return float(np.mean(epoch_losses))

    def evaluate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the reconstruction error for each sample.
        Returns a 1D numpy array representing sample losses.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            reconstructed, _ = self.model(X_tensor)
            
            # Compute element-wise squared error, then mean per row (sample)
            squared_errors = torch.pow(reconstructed - X_tensor, 2)
            mean_squared_errors = torch.mean(squared_errors, dim=1)
            
            return mean_squared_errors.cpu().numpy()

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Extract the lower dimensional latent features of the sensors.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            _, latent = self.model(X_tensor)
            return latent.cpu().numpy()
