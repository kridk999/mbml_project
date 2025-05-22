import torch
import torch.nn as nn
from entmax import entmax15


class LSTMWinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0, bidirectional=False, attention=False):
        """
        Args:
            input_dim (int): Number of input features per round.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
            bidirectional (bool): Whether to use a bidirectional LSTM.
            attention (bool): Whether to use attention mechanism.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_directions = 2 if bidirectional else 1

        # Initial feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        if attention:
            self.attention_weights = nn.Linear(hidden_dim * self.num_directions, 1)

        # Output layers
        lstm_output_dim = hidden_dim * self.num_directions
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 2),  # Binary classification with 2 outputs
        )

    def apply_attention(self, lstm_output):
        """Apply attention mechanism to LSTM output."""
        # Calculate attention weights
        attn_weights = torch.softmax(self.attention_weights(lstm_output), dim=1)

        # Apply weights to LSTM output
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)

        # Expand context vector to the same size as lstm_output
        expanded_context = context.expand_as(lstm_output)

        # Concatenate context with lstm_output
        attended_output = lstm_output + expanded_context

        return attended_output

    def forward(self, x, initial_states=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            initial_states: Optional tuple of (h0, c0) hidden states

        Returns:
            torch.Tensor: Output probability vector of shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project features
        x = self.feature_projection(x)

        # Pass through LSTM
        if initial_states is not None:
            lstm_out, _ = self.lstm(x, initial_states)
        else:
            lstm_out, _ = self.lstm(x)

        # Apply attention if enabled
        if self.attention:
            lstm_out = self.apply_attention(lstm_out)

        # Pass through output layers
        logits = self.output_layers(lstm_out)

        # Apply entmax15 for sparse probability distribution
        batch_size, seq_len, output_dim = logits.shape
        logits_reshaped = logits.reshape(-1, output_dim)
        probs_reshaped = entmax15(logits_reshaped, dim=-1)
        probs = probs_reshaped.reshape(batch_size, seq_len, output_dim)

        # Return probability of positive class (team1 winning)
        return probs[:, :, 1]


class ResponsiveWinPredictor(nn.Module):
    """
    Enhanced model that's more sensitive to critical events during a round.

    Features:
    - Initial state based on starting conditions (equipment values)
    - Separate processing paths for different feature types
    - Event-detection subnet for high-impact events
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        player_count_indices=None,  # Indices of features related to player counts
        equipment_indices=None,  # Indices of features related to equipment
        use_entmax=True,  # Whether to use entmax activation (vs sigmoid)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.player_count_indices = player_count_indices or []
        self.equipment_indices = equipment_indices or []
        self.use_entmax = use_entmax

        # Initial state projection (for equipment values)
        self.initial_state_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        # Main LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Event detection subnet - focused on player count changes
        if self.player_count_indices:
            player_count_dim = len(self.player_count_indices)
            self.event_detector = nn.Sequential(
                nn.Linear(player_count_dim * 2, hidden_dim // 2),  # Current + previous values
                nn.LeakyReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.LeakyReLU(),
            )

        # Output projection
        combined_dim = hidden_dim
        if self.player_count_indices:
            combined_dim += hidden_dim // 2

        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 if use_entmax else 1),  # 2 outputs for entmax, 1 for sigmoid
        )

    def extract_subfeatures(self, x, indices):
        """Extract specific feature dimensions from input tensor."""
        if not indices:
            return None
        return x[:, :, indices]

    def forward(self, x):
        """
        Forward pass with enhanced sensitivity to game events.

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
