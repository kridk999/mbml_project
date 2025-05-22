import torch
import torch.nn as nn
from entmax import entmax15


class LSTMWinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        """
        Args:
            input_dim (int): Number of input features per round.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability (only applies if num_layers > 1).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Fully connected layer to output a scalar for each round.
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, round_length, input_dim)

        Returns:
            torch.Tensor: Output probability vector of shape (batch_size, round_length)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, round_length, hidden_dim)
        logits = self.fc(lstm_out).squeeze(-1)  # logits: (batch_size, round_length)
        probs = self.sigmoid(logits)  # probs: (batch_size, round_length)
        return probs


# Example usage:
if __name__ == "__main__":
    # Dummy data: a batch of 2 matches, each with 10 rounds and 5 features per round.
    batch_size = 2
    round_length = 10
    input_dim = 5
    dummy_input = torch.randn(batch_size, round_length, input_dim)

    model = LSTMWinPredictor(input_dim=input_dim, hidden_dim=32, num_layers=2, dropout=0.2)
    output_probs = model(dummy_input)
    print("Output shape:", output_probs.shape)  # Expected shape: (2, 10)
    print("Output probabilities:", output_probs)
