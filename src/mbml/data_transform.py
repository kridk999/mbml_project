import numpy as np
import torch


class FeatureEmphasis:
    """
    Transformer to emphasize important features like player counts and equipment values.

    This helps the model learn to react more strongly to changes in these critical values.
    """

    def __init__(self, player_count_emphasis=2.0, equipment_emphasis=1.5, player_count_cols=None, equipment_cols=None):
        """
        Args:
            player_count_emphasis: Multiplier for player count features
            equipment_emphasis: Multiplier for equipment value features
            player_count_cols: List of column indices for player count features
            equipment_cols: List of column indices for equipment value features
        """
        self.player_count_emphasis = player_count_emphasis
        self.equipment_emphasis = equipment_emphasis

        # Default column indices if not provided
        self.player_count_cols = player_count_cols or []
        self.equipment_cols = equipment_cols or []

    def __call__(self, features):
        """
        Apply emphasis to specific features.

        Args:
            features: Tensor of shape (seq_len, num_features)

        Returns:
            Transformed features tensor of same shape
        """
        emphasized_features = features.clone()

        # Apply emphasis to player count features
        for col in self.player_count_cols:
            emphasized_features[:, col] *= self.player_count_emphasis

        # Apply emphasis to equipment value features
        for col in self.equipment_cols:
            emphasized_features[:, col] *= self.equipment_emphasis

        return emphasized_features


class DeltaFeatures:
    """
    Add change/delta features to help the model identify important changes.

    This highlights changes in key metrics like player counts or equipment values.
    """

    def __init__(self, key_feature_indices=None):
        """
        Args:
            key_feature_indices: List of feature indices to create deltas for
        """
        self.key_feature_indices = key_feature_indices or []

    def __call__(self, features):
        """
        Add delta features to the input tensor.

        Args:
            features: Tensor of shape (seq_len, num_features)

        Returns:
            Enhanced features tensor with added delta columns
        """
        seq_len, num_features = features.shape

        # No deltas possible for sequences of length 1
        if seq_len <= 1:
            # Just pad with zeros for consistent output shape
            delta_features = torch.zeros((seq_len, len(self.key_feature_indices)), dtype=features.dtype)
            return torch.cat([features, delta_features], dim=1)

        # Create new feature tensor with original features plus deltas
        new_features = torch.zeros((seq_len, num_features + len(self.key_feature_indices)), dtype=features.dtype)

        # Copy original features
        new_features[:, :num_features] = features

        # Calculate deltas for specified features
        for i, idx in enumerate(self.key_feature_indices):
            # Compute differences (t - (t-1))
            deltas = features[1:, idx] - features[:-1, idx]

            # Pad first entry with 0 since there's no previous frame
            padded_deltas = torch.cat([torch.zeros(1, dtype=features.dtype), deltas])

            # Add delta feature
            new_features[:, num_features + i] = padded_deltas

        return new_features


class FeatureNormalizer:
    """
    Normalize features to help the model learn more effectively.

    Different normalization strategies can be applied to different feature groups.
    """

    def __init__(self):
        self.means = None
        self.stds = None
        self.mins = None
        self.maxs = None

    def fit(self, features_batch):
        """
        Calculate normalization parameters from a batch of features.

        Args:
            features_batch: Tensor of shape (batch_size, seq_len, num_features)
        """
        # Reshape to combine batch and sequence dimensions
        combined = features_batch.view(-1, features_batch.shape[-1])

        # Calculate statistics
        self.means = combined.mean(dim=0)
        self.stds = combined.std(dim=0)
        self.stds[self.stds == 0] = 1.0  # Avoid division by zero

        self.mins = combined.min(dim=0)[0]
        self.maxs = combined.max(dim=0)[0]
        self.ranges = self.maxs - self.mins
        self.ranges[self.ranges == 0] = 1.0  # Avoid division by zero

    def normalize(self, features, method="zscore"):
        """
        Normalize features using fitted parameters.

        Args:
            features: Tensor of shape (seq_len, num_features)
            method: 'zscore' or 'minmax'

        Returns:
            Normalized features
        """
        if self.means is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if method == "zscore":
            return (features - self.means) / self.stds
        elif method == "minmax":
            return (features - self.mins) / self.ranges
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def __call__(self, features):
        """
        Apply normalization to features.

        Args:
            features: Tensor of shape (seq_len, num_features)

        Returns:
            Normalized features
        """
        return self.normalize(features, method="zscore")
