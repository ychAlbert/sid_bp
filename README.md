# A Unified Comparison Framework for Backpropagation-Free Algorithms

This repository provides a unified and modular framework for training and comparing multiple Backpropagation-Free (BP-Free) deep learning algorithms. By integrating several prominent methods into a single workflow, this framework enables researchers to conduct fair experimental comparisons, share datasets, and manage results with ease.

Currently, the following algorithms are integrated:
- **[SID (Our Method)]** and a **Baseline** using standard backpropagation.
- [**Synthetic Gradients (SG / DNI)**](https://arxiv.org/abs/1608.05343)
- [**The Forward-Forward Algorithm (FF)**](https://www.cs.toronto.edu/~hinton/FFA13.pdf)
- [**Equilibrium Propagation (EP)**](https://arxiv.org/abs/2006.03824)

## ✨ Key Features

- **Unified Training Entrypoint**: A single `main.py` script to run, configure, and evaluate any integrated algorithm.
- **Shared Data and Results**: All algorithms use a common `./data` directory for datasets and save outputs to a structured `./result` directory for easy management and analysis.
- **Modular Algorithm Design**: Each algorithm is encapsulated in a dedicated wrapper, decoupling it from the main framework for easy maintenance and extension.
- **Flexible Configuration**: Supports both common command-line arguments (e.g., `epochs`, `batch_size`) and algorithm-specific hyperparameters via JSON strings.

## 📁 Project Structure

```
.
├── main.py                 # Unified entrypoint for all experiments
├── sid.py                  # Implementation of the SID algorithm and the baseline
├── README.md               # This document
├── requirements.txt        # Project dependencies
│
├── data/                     # Shared directory for datasets (auto-created)
│   ├── mnist/
│   └── cifar10/
│
├── result/                   # Shared directory for experiment results (auto-created)
│   ├── sid/
│   ├── baseline/
│   ├── synthetic_gradient/
│   ├── forward_forward/
│   └── equilibrium_propagation/
│
└── algorithms/               # Directory for all third-party algorithm implementations
    ├── __init__.py
    ├── sg_wrapper.py         # Wrapper for Synthetic Gradients
    ├── ff_wrapper.py         # Wrapper for Forward-Forward
    ├── ep_wrapper.py         # Wrapper for Equilibrium Propagation
    │
    ├── synthetic_gradient/   # Full source code for SG
    ├── forward_forward/      # Full source code for FF
    └── equilibrium_propagation/ # Full source code for EP
```

## 🚀 Installation and Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    # Using conda
    conda create -n bp_free python=3.8
    conda activate bp_free

    # Or using venv
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` file should contain libraries like `torch`, `torchvision`, `matplotlib`, etc.)*

4.  **Prepare Datasets**
    No manual download is needed. The framework automatically downloads and saves datasets from `torchvision` to the `./data` directory the first time they are used.

## 💡 How to Use

All experiments are launched via `main.py`. The primary argument is `--algorithm`, which selects the model to run.

### Basic Usage

```bash
python main.py --algorithm <algorithm_name> [COMMON_OPTIONS]
```

**Common Options:**

- `--algorithm`: **(Required)** The algorithm to run. Choices: `sid`, `baseline`, `synthetic_gradient`, `forward_forward`, `equilibrium_propagation`.
- `--dataset`: The dataset to use. Default: `cifar10`. Choices: `mnist`, `cifar10`.
- `--epochs`: Number of training epochs. Default: `100`.
- `--batch_size`: Training batch size. Default: `128`.
- `--learning_rate`: The learning rate. Default: `0.01`. (Note: This may be overridden by more complex schedulers in some algorithms like EP).
- `--device`: The device to run on. Default: `cuda` if available, otherwise `cpu`.

### Specifying Algorithm-Specific Arguments

Each algorithm has unique hyperparameters that can be passed as a JSON-formatted string using the `--alg_args` argument.

**Important:** Pay close attention to quoting when passing JSON from the command line!
-   On **Linux, macOS, or Git Bash**, use single quotes `'` to wrap the JSON string:
    ```bash
    --alg_args '{"key": "value", "number": 123}'
    ```
-   On **Windows Command Prompt (cmd.exe)**, use double quotes `"` to wrap the string and escape the inner double quotes with a backslash `\`:
    ```cmd
    --alg_args "{\"key\": \"value\", \"number\": 123}"
    ```

---

## 🔬 Examples

Here are sample commands for running each algorithm.

### 1. SID (Our Method) and Baseline

*(Please add specific parameters and examples based on your `sid.py` implementation.)*

```bash
# Run the SID algorithm on CIFAR-10
python main.py --algorithm sid --dataset cifar10 --epochs 150

# Run the backpropagation baseline
python main.py --algorithm baseline --dataset cifar10 --epochs 150
```

### 2. Synthetic Gradients (SG)

**Key configurable arguments (`--alg_args`):**
- `model_type` (str): Model architecture, `cnn` or `mlp`. Default: `cnn`.
- `conditioned` (bool): Whether to use conditioned Decoupled Neural Interfaces (DNI). Default: `false`.

**Examples:**
```bash
# Run the CNN model on CIFAR-10 with default settings
python main.py --algorithm synthetic_gradient --dataset cifar10 --epochs 300 --batch_size 100

# Run a conditioned MLP model on MNIST
python main.py --algorithm synthetic_gradient --dataset mnist --epochs 100 \
--alg_args '{"model_type": "mlp", "conditioned": true}'
```

### 3. Forward-Forward (FF)

**Key configurable arguments (`--alg_args`):**
- `model` (dict):
    - `hidden_dim` (int): Dimension of hidden layers. Default: `1000`.
    - `num_layers` (int): Number of layers. Default: `3`.
- `training` (dict):
    - `downstream_learning_rate` (float): Learning rate for the final classification head. Default: `1e-2`.

**Examples:**
```bash
# Run with default parameters on CIFAR-10
python main.py --algorithm forward_forward --dataset cifar10 --epochs 100 --batch_size 100

# Customize the model architecture
python main.py --algorithm forward_forward --dataset cifar10 --epochs 100 \
--alg_args '{"model": {"num_layers": 4, "hidden_dim": 2048}}'
```

### 4. Equilibrium Propagation (EP)

EP has numerous parameters; the wrapper uses default values recommended in the original paper.

**Key configurable arguments (`--alg_args`):**
- `lrs` (list): A list of layer-wise learning rates.
- `betas` (list): EP beta parameters `[beta_1, beta_2]`. Default: `[0.0, 0.5]`.
- `T1`, `T2` (int): Number of iterations for the free and nudged phases. Default: `T1=250`, `T2=30`.
- `loss` (str): Loss function, `mse` or `cel`. Default: `mse`.

**Examples:**
```bash
# Run with recommended defaults for CIFAR-10
python main.py --algorithm equilibrium_propagation --dataset cifar10 --epochs 120 --batch_size 128

# Modify beta values and switch to CrossEntropy loss
python main.py --algorithm equilibrium_propagation --dataset cifar10 --epochs 120 \
--alg_args '{"betas": [0.0, 1.0], "loss": "cel", "softmax": true}'
```

## 📊 Reviewing Results

All experiment outputs, including log files, model checkpoints (`.pkl` or `.pt`), and performance plots, are saved in the `result/` directory, organized into subfolders named after the algorithm.

## 🧩 How to Add a New Algorithm

The framework's modular design makes it straightforward to integrate a new BP-Free algorithm:

1.  **Add Source Code**: Place the complete source code for the new algorithm into its own folder inside `algorithms/`.
2.  **Create `__init__.py`**: Add an empty `__init__.py` file in the new algorithm's root folder to mark it as a Python package.
3.  **Write a Wrapper**: Create a new `new_alg_wrapper.py` file in the `algorithms/` directory.
4.  **Implement the `run` Function**: Inside the wrapper, implement a function named `run(config)`. This function is responsible for:
    -   Receiving the unified `config` dictionary.
    -   Translating `config` into the parameter format required by the new algorithm (e.g., `argparse.Namespace`).
    -   Ensuring the algorithm loads data from `config['data_path']`.
    -   Ensuring the algorithm saves results to `config['result_path']`.
    -   Calling the core training logic of the new algorithm.
5.  **Integrate into Entrypoint**: In `main.py`, import your new wrapper and add an `elif` branch to the main logic to call its `run` function.
