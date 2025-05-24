#!/usr/bin/env python3
"""
Responsive Bayesian LSTM
========================

Combines the event-sensitivity of the responsive model with the uncertainty
quantification of the Bayesian approach.

Key improvements:
- Event detection subnet for player count changes
- Separate processing paths for different feature types
- Initial state projection from equipment values
- Bayesian treatment of weights for uncertainty quantification
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from entmax import entmax15


class ResponsiveBayesianLSTM(PyroModule):
    """
    Bayesian LSTM with improved responsiveness to game events.
    
    This model combines:
    1. The event-sensitivity features from ResponsiveWinPredictor
    2. Uncertainty quantification through Bayesian treatment of parameters
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        player_count_indices=None,  # Indices of features related to player counts
        equipment_indices=None,     # Indices of features related to equipment
        use_entmax=True,            # Whether to use entmax activation (vs sigmoid)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.player_count_indices = player_count_indices or []
        self.equipment_indices = equipment_indices or []
        self.use_entmax = use_entmax
        
        # Initial state projection (for equipment values)
        self.initial_state_projection = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](input_dim, hidden_dim),
            PyroModule[nn.Tanh]()
        )
        
        # Main LSTM for sequence processing
        self.lstm = PyroModule[nn.LSTM](
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Event detection subnet - focused on player count changes
        if self.player_count_indices:
            player_count_dim = len(self.player_count_indices)
            self.event_detector = PyroModule[nn.Sequential](
                PyroModule[nn.Linear](player_count_dim * 2, hidden_dim // 2),  # Current + previous values
                PyroModule[nn.LeakyReLU](),
                PyroModule[nn.Linear](hidden_dim // 2, hidden_dim // 2),
                PyroModule[nn.LeakyReLU](),
            )
        
        # Output projection
        combined_dim = hidden_dim
        if self.player_count_indices:
            combined_dim += hidden_dim // 2
        
        self.output_projection = PyroModule[nn.Sequential](
            PyroModule[nn.Linear](combined_dim, hidden_dim),
            PyroModule[nn.LeakyReLU](),
            PyroModule[nn.Dropout](dropout),
            PyroModule[nn.Linear](hidden_dim, 2 if use_entmax else 1),  # 2 outputs for entmax, 1 for sigmoid
        )
    
    def extract_subfeatures(self, x, indices):
        """Extract specific feature dimensions from input tensor."""
        if not indices:
            return None
        return x[:, :, indices]
    
    def forward(self, x):
        """
        Forward pass with Bayesian uncertainty and enhanced sensitivity to game events.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output probability vector of shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create initial state based on first frame equipment values
        if self.equipment_indices:
            equipment_values = self.extract_subfeatures(x, self.equipment_indices)
            initial_state = self.initial_state_projection(x[:, 0])
            h0 = initial_state.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            initial_states = (h0, c0)
        else:
            initial_states = None
        
        # Main LSTM processing
        lstm_out, _ = self.lstm(x, initial_states)
        
        # Event detection for player counts
        event_features = None
        if self.player_count_indices:
            player_counts = self.extract_subfeatures(x, self.player_count_indices)
            
            # Get current and previous player counts
            current_counts = player_counts
            prev_counts = torch.cat([player_counts[:, :1], player_counts[:, :-1]], dim=1)
            
            # Concatenate for delta detection
            player_deltas = torch.cat([current_counts, prev_counts], dim=2)
            
            # Process through event detector
            event_features = self.event_detector(player_deltas)
        
        # Combine features
        if event_features is not None:
            combined_features = torch.cat([lstm_out, event_features], dim=2)
        else:
            combined_features = lstm_out
        
        # Output projection
        logits = self.output_projection(combined_features)
        
        # Apply activation
        if self.use_entmax:
            # Reshape for entmax
            batch_size, seq_len, output_dim = logits.shape
            logits_reshaped = logits.reshape(-1, output_dim)
            probs_reshaped = entmax15(logits_reshaped, dim=-1)
            probs = probs_reshaped.reshape(batch_size, seq_len, output_dim)
            return probs[:, :, 1]  # Return probability of positive class
        else:
            # Apply sigmoid for binary output
            return torch.sigmoid(logits.squeeze(-1))
    
    def model(self, x, y=None):
        """
        Pyro model - defines the probabilistic generative process.
        """
        # Register modules with Pyro
        pyro.module("initial_state_projection", self.initial_state_projection)
        pyro.module("lstm", self.lstm)
        
        if hasattr(self, "event_detector"):
            pyro.module("event_detector", self.event_detector)
            
        pyro.module("output_projection", self.output_projection)
        
        # Get predictions from forward pass
        predictions = self.forward(x)
        
        # Simple likelihood - sample observations where we have valid data
        if y is not None:
            batch_size, seq_len = predictions.shape
            for i in range(batch_size):
                for j in range(seq_len):
                    # Skip positions with NaN labels
                    if torch.isnan(y[i, j]):
                        continue
                        
                    pyro.sample(
                        f"obs_{i}_{j}", 
                        dist.Bernoulli(predictions[i, j]), 
                        obs=y[i, j]
                    )
        
        return predictions
    
    def guide(self, x, y=None):
        """
        Pyro guide - defines the variational approximation.
        For simplicity, just register the model parameters.
        """
        pyro.module("initial_state_projection", self.initial_state_projection)
        pyro.module("lstm", self.lstm)
        
        if hasattr(self, "event_detector"):
            pyro.module("event_detector", self.event_detector)
            
        pyro.module("output_projection", self.output_projection)
