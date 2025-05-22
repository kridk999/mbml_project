# CS:GO Round Win Prediction with LSTM

An LSTM-based model for predicting round outcomes in Counter-Strike: Global Offensive matches based on sequential round data. The model takes round frame data and outputs win probabilities for each frame, enabling real-time win probability estimation.

## Project Description

This project implements a Long Short-Term Memory (LSTM) neural network model to predict the probability of a team winning a round in Counter-Strike: Global Offensive (CS:GO) matches. The model processes sequential data from round frames and outputs a probability vector for each timestep in the round.

### Key Features
- Variable-length sequence handling with proper padding
- Real-time win probability estimation for each frame in a round
- Custom PyTorch datasets and dataloaders for efficient data processing

## Project Structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── mbml/                 # Main package
│   │   ├── __init__.py
│   │   ├── model.py         # LSTM model architecture
│   │   ├── dataset.py       # Custom dataset and dataloaders
│   │   ├── evaluate.py      # Model evaluation utilities
│   │   ├── train.py         # Training loop implementation
│   │   └── predict.py       # Inference and visualization
├── examples/                 # Example scripts
│   └── lstm_example.py      # Example of using the model
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


## Model Architecture

The `LSTMWinPredictor` model consists of:
- An LSTM module that processes sequential round data
- A fully-connected layer that transforms LSTM outputs to logits
- A sigmoid activation function that converts logits to probabilities

The model takes input data with shape `(batch_size, round_length, input_dim)` and outputs probabilities with shape `(batch_size, round_length)`.

## Data Format

The dataset is structured as a CSV file with multiple frames per round. Each frame contains features related to the current state of the round, such as:
- Equipment values
- Player counts
- Bomb status
- Utility counts
- And more

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mbml_project

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Training

To train the model:

```bash
python run.py --mode train --csv_path round_frame_data.csv
```

### Prediction

To predict with a trained model:

```bash
python run.py --mode predict --csv_path round_frame_data.csv --model_path models/best_model.pt
```

### Training and Prediction

To train and then predict:

```bash
python run.py --mode train_and_predict --csv_path round_frame_data.csv
```

### Additional Arguments

- `--hidden_dim`: Hidden dimension of the LSTM (default: 64)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout probability (default: 0.2)
- `--batch_size`: Batch size for training/inference (default: 32)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model_dir`: Directory to save model files
- `--output_dir`: Directory to save prediction results
- `--num_rounds`: Number of rounds to predict (default: all)
- `--cpu`: Force CPU usage even if CUDA is available

## Example

An example script is provided in `examples/lstm_example.py`:

```bash
python examples/lstm_example.py
```

A Jupyter notebook with a complete workflow is also available in `notebooks/lstm_round_win_prediction.ipynb`.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

---

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
