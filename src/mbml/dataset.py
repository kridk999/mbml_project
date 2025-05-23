import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CSGORoundDataset(Dataset):
    """
    Dataset for CS:GO round data that groups frames by match_id and round_idx.

    Each item represents a complete round from a match with all its frames/timesteps.
    """

    def __init__(
        self,
        csv_path: str,
        team1_col: str = "ctTeam",  # Column name containing the team1 name
        team2_col: str = "tTeam",  # Column name containing the team2 name
        winner_col: str = "rnd_winner",  # Column name containing the round winner (CT or T)
        match_id_col: str = "match_id",
        round_idx_col: str = "round_idx",
        feature_cols: Optional[List[str]] = None,
        transform=None,
        preload: bool = True,
    ):
        """
        Args:
            csv_path: Path to the CSV data file
            team1_col: Column name for team1 (by default, CT team)
            team2_col: Column name for team2 (by default, T team)
            winner_col: Column name for round winner
            match_id_col: Column name for match ID
            round_idx_col: Column name for round index
            feature_cols: List of column names to use as features. If None, will use sensible defaults
            transform: Optional transform to apply to the features
            preload: If True, preload all data into memory
        """
        self.csv_path = csv_path
        self.team1_col = team1_col
        self.team2_col = team2_col
        self.winner_col = winner_col
        self.match_id_col = match_id_col
        self.round_idx_col = round_idx_col
        self.transform = transform

        # Read the CSV file
        print(f"Loading data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        # If feature_cols is None, use all numerical columns except specific ones
        if feature_cols is None:
            # Exclude these columns from features
            exclude_cols = [
                match_id_col,
                round_idx_col,
                team1_col,
                team2_col,
                winner_col,
                "clock",  # Exclude 'clock' as it's not numerical
            ]
            # Get all numerical columns
            numerical_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            self.feature_cols = [col for col in numerical_cols if col not in exclude_cols]

            # Print warning if we detect a potential mismatch with saved models
            if len(self.feature_cols) != 12:
                print(f"WARNING: Found {len(self.feature_cols)} features, but trained models may expect exactly 12.")
                print(
                    "If you encounter dimension mismatch errors, "
                    "specify exact feature columns when creating the dataset."
                )
        else:
            self.feature_cols = feature_cols

        print(f"Using features: {self.feature_cols}")

        # Get unique (match_id, round_idx) combinations
        self.rounds = (
            self.df.groupby([match_id_col, round_idx_col])
            .size()
            .reset_index()[[match_id_col, round_idx_col]]
            .values.tolist()
        )

        print(f"Found {len(self.rounds)} unique rounds")

        # Preload data if requested
        self.preloaded_data = {}
        if preload:
            print("Preloading data...")
            for i, (match_id, round_idx) in enumerate(self.rounds):
                if i % 1000 == 0:
                    print(f"Preloaded {i}/{len(self.rounds)} rounds")
                # Get the slice of the DataFrame for this round
                round_df = self.df[(self.df[match_id_col] == match_id) & (self.df[round_idx_col] == round_idx)]

                # Extract features and labels
                features, labels = self._extract_features_labels(round_df)
                self.preloaded_data[(match_id, round_idx)] = (features, labels)

            print(f"Preloaded {len(self.preloaded_data)} rounds")

    def _extract_features_labels(self, round_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and labels from a round DataFrame.

        Args:
            round_df: DataFrame containing frames for a single round

        Returns:
            Tuple of (features, labels)
            - features: Tensor of shape (seq_len, num_features)
            - labels: Tensor of shape (seq_len,) with binary values indicating if team1 wins
        """
        # Extract features
        features = torch.tensor(round_df[self.feature_cols].values, dtype=torch.float32)

        # Create labels: 1 if T side wins, 0 if CT side wins
        # Note: We replicate the label for each frame in the round
        # The CSV has 'CT' or 'T' in the winner_col
        winner = round_df.iloc[0][self.winner_col]
        
        # Directly map the winner to a label: T win = 1, CT win = 0
        # This makes the prediction consistent regardless of team names
        if winner == "T":
            # T team won - label is 1
            label = 1.0
        else:
            # CT team won - label is 0
            label = 0.0

        # Replicate the label for each frame in the round
        labels = torch.full((len(round_df),), label, dtype=torch.float32)

        # Apply transform if provided
        if self.transform:
            features = self.transform(features)

        return features, labels

    def __len__(self) -> int:
        """Return the number of rounds in the dataset."""
        return len(self.rounds)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Tuple[str, int]]]:
        """
        Get a round from the dataset.

        Args:
            idx: Index of the round

        Returns:
            Dictionary with keys:
                'features': Tensor of shape (seq_len, num_features)
                'labels': Tensor of shape (seq_len,) with binary values indicating if team1 wins
                'metadata': Tuple of (match_id, round_idx)
        """
        match_id, round_idx = self.rounds[idx]

        if hasattr(self, "preloaded_data") and (match_id, round_idx) in self.preloaded_data:
            # Use preloaded data
            features, labels = self.preloaded_data[(match_id, round_idx)]
        else:
            # Load data on-the-fly
            round_df = self.df[(self.df[self.match_id_col] == match_id) & (self.df[self.round_idx_col] == round_idx)]
            features, labels = self._extract_features_labels(round_df)

        return {"features": features, "labels": labels, "metadata": (match_id, round_idx)}


def collate_variable_length_rounds(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length rounds.

    This function:
    1. Pads sequences to the maximum length in the batch
    2. Creates a mask indicating valid (non-padded) positions
    3. Stacks features, labels, and metadata

    Args:
        batch: List of dictionaries, each containing 'features', 'labels', and 'metadata'

    Returns:
        Dictionary with keys:
            'features': Tensor of shape (batch_size, max_seq_len, num_features) with padding
            'labels': Tensor of shape (batch_size, max_seq_len) with padding
            'mask': Tensor of shape (batch_size, max_seq_len) with 1s for valid positions and 0s for padding
            'metadata': List of (match_id, round_idx) tuples
    """
    # Get sequence lengths
    seq_lengths = [item["features"].shape[0] for item in batch]
    max_seq_len = max(seq_lengths)

    # Get the feature dimensionality
    feature_dim = batch[0]["features"].shape[1]

    # Initialize tensors
    batch_size = len(batch)
    features = torch.zeros(batch_size, max_seq_len, feature_dim)
    labels = torch.zeros(batch_size, max_seq_len)
    mask = torch.zeros(batch_size, max_seq_len)
    metadata = []

    # Fill in the tensors
    for i, item in enumerate(batch):
        seq_len = item["features"].shape[0]
        features[i, :seq_len, :] = item["features"]
        labels[i, :seq_len] = item["labels"]
        mask[i, :seq_len] = 1.0  # Mark valid positions
        metadata.append(item["metadata"])

    return {"features": features, "labels": labels, "mask": mask, "metadata": metadata}


def create_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 4,
    chronological: bool = False,  # New parameter to control chronological splitting
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        csv_path: Path to the CSV data file
        batch_size: Batch size for DataLoaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        random_seed: Random seed for reproducibility
        num_workers: Number of workers for DataLoader
        chronological: If True, splits data chronologically to avoid look-ahead bias
        **dataset_kwargs: Additional arguments for CSGORoundDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create the dataset
    dataset = CSGORoundDataset(csv_path, **dataset_kwargs)

    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    if chronological:
        # For chronological splitting, we assume the rounds are already in chronological order
        # or we need to sort them based on match time (if available)
        indices = list(range(dataset_size))

        # Split indices chronologically
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        print(f"Dataset split chronologically: {train_size} train, {val_size} val, {test_size} test")
    else:
        # Random split (original behavior)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        print(f"Dataset split randomly: {train_size} train, {val_size} val, {test_size} test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not chronological,  # Only shuffle if not using chronological order
        num_workers=num_workers,
        collate_fn=collate_variable_length_rounds,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_length_rounds,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_length_rounds,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Example: Create dataset and dataloaders from CSV
    csv_path = "round_frame_data.csv"

    # Try loading with a small sample to test
    df = pd.read_csv(csv_path, nrows=100)
    print(f"Sample data columns: {df.columns.tolist()}")
    print(f"Sample data shape: {df.shape}")

    # Create dataloaders with chronological splitting to avoid look-ahead bias
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        batch_size=32,
        preload=True,  # Set to False for very large datasets
        chronological=True,  # Enable chronological splitting
    )

    # Test a batch
    for batch in train_loader:
        features = batch["features"]
        labels = batch["labels"]
        mask = batch["mask"]
        metadata = batch["metadata"]

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Number of metadata items: {len(metadata)}")
        print(f"First metadata item: {metadata[0]}")
        break
