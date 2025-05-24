import torch
from responsive_bayesian_lstm import ResponsiveBayesianLSTM
import pyro

# 1. Instantiate the model
input_dim = 8       # Based on DEFAULT_FEATURE_COLS in train script
hidden_dim = 64
seq_len = 10        # Example sequence length for demonstration

# Create the model with some reasonable defaults
model = ResponsiveBayesianLSTM(
    input_dim=input_dim, 
    hidden_dim=hidden_dim,
    player_count_indices=[1, 2],  # numCTAlive, numTAlive
    equipment_indices=[3, 4]      # ctEquipValue, tEquipValue
)

# 2. Create sample input tensors with the correct shapes
batch_size = 1
sample_x = torch.zeros(batch_size, seq_len, input_dim)
sample_y = torch.zeros(batch_size, seq_len)

# 3. Render the PGM to disk
try:
    # Try with simpler arguments first
    pyro.render_model(
        model.model, 
        model.guide,
        (sample_x, sample_y),
        render_distributions=True,
        filename="responsive_bayesian_pgm.png"
    )
    print("✅ PGM visualization saved as 'responsive_bayesian_pgm.png'")
except Exception as e:
    print(f"❌ Error rendering PGM: {e}")
    
    # Try alternative approach with daft for PGM visualization
    try:
        import daft
        print("🔄 Trying alternative PGM visualization with daft...")
        
        # Create a simple PGM diagram showing the model structure
        pgm = daft.PGM([6, 4], origin=[0, 0])
        
        # Add nodes for the Bayesian LSTM components
        pgm.add_node(daft.Node("x_t", r"$x_t$", 1, 3, observed=True))
        pgm.add_node(daft.Node("h_t", r"$h_t$", 2, 3))
        pgm.add_node(daft.Node("theta", r"$\theta$", 3, 2))
        pgm.add_node(daft.Node("y_t", r"$y_t$", 4, 3, observed=True))
        pgm.add_node(daft.Node("event", r"$e_t$", 2, 1))
        
        # Add edges
        pgm.add_edge("x_t", "h_t")
        pgm.add_edge("h_t", "y_t")
        pgm.add_edge("theta", "h_t")
        pgm.add_edge("theta", "y_t")
        pgm.add_edge("event", "h_t")
        
        # Render and save
        pgm.render()
        pgm.savefig("responsive_bayesian_pgm_daft.png", dpi=150)
        print("✅ Alternative PGM visualization saved as 'responsive_bayesian_pgm_daft.png'")
        
    except ImportError:
        print("💡 daft library not available. Install with: pip install daft")
        print("🔧 Creating a simple text-based model description instead...")
        
        # Create a text-based description
        model_description = """
Responsive Bayesian LSTM Model Structure:
========================================

Input: x_t (batch_size, seq_len, input_dim)
│
├─► Initial State Projection (equipment values)
│   └─► h_0, c_0
│
├─► Main LSTM Processing
│   ├─► input_dim → hidden_dim
│   └─► outputs: lstm_out (batch_size, seq_len, hidden_dim)
│
├─► Event Detection Subnet (player counts)
│   ├─► Current + Previous player counts
│   ├─► hidden_dim//2 → hidden_dim//2
│   └─► outputs: event_features (batch_size, seq_len, hidden_dim//2)
│
└─► Output Projection
    ├─► combined_features: [lstm_out, event_features]
    ├─► combined_dim → hidden_dim → 2 (entmax) or 1 (sigmoid)
    └─► outputs: y_t (batch_size, seq_len)

Bayesian Components:
- All linear layers wrapped with PyroModule
- Probabilistic treatment via pyro.module()
- Uncertainty quantification through SVI
        """
        
        with open("responsive_bayesian_model_structure.txt", "w") as f:
            f.write(model_description)
        print("✅ Model structure saved as 'responsive_bayesian_model_structure.txt'")
        
    except Exception as daft_error:
        print(f"❌ Daft visualization also failed: {daft_error}")
        print("📋 Check that graphviz and/or daft are properly installed")