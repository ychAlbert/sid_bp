# 统一化BP-Free算法比较框架

本项目旨在提供一个统一、模块化的框架，用于训练和比较多种反向传播-无关（Backpropagation-Free）的深度学习算法。通过将多个独立的算法仓库整合到一个统一的入口和工作流下，研究人员可以更便捷地进行公平的实验对比、共享数据集和管理实验结果。

目前集成的算法包括：
- **SID (本文方法)** 和 **Baseline (基线)**
- [**Synthetic Gradients (SG)**](https://arxiv.org/abs/1608.05343)
- [**The Forward-Forward Algorithm (FF)**](https://www.cs.toronto.edu/~hinton/FFA13.pdf)
- [**Equilibrium Propagation (EP)**](https://arxiv.org/abs/2006.03824)

## ✨ 主要特性

- **统一的训练入口**: 使用单个 `main.py` 脚本即可启动任何已集成的算法。
- **共享的数据与结果目录**: 所有算法共用 `./data` 目录下的数据集，并将实验结果统一输出到 `./result` 目录，便于管理和分析。
- **模块化算法设计**: 每个算法都被封装在一个独立的"包装器"（wrapper）中，使其与主框架解耦，易于维护和扩展。
- **灵活的参数配置**: 支持通过命令行传递通用参数（如 `epochs`, `batch_size`）和特定于算法的复杂超参数（通过JSON字符串）。

## 📁 项目结构

```
.
├── main.py                 # 统一的实验启动入口
├── sid.py                  # SID算法和基线方法的实现
├── README.md               # 本文档
├── requirements.txt        # 项目依赖
│
├── data/                     # 共享的数据集目录 (自动创建和下载)
│   ├── mnist/
│   └── cifar10/
│
├── result/                   # 共享的实验结果目录 (自动创建)
│   ├── sid/
│   ├── baseline/
│   ├── synthetic_gradient/
│   ├── forward_forward/
│   └── equilibrium_propagation/
│
└── algorithms/               # 存放所有移植的第三方算法
    ├── __init__.py
    ├── sg_wrapper.py         # SG 的包装器
    ├── ff_wrapper.py         # FF 的包装器
    ├── ep_wrapper.py         # EP 的包装器
    │
    ├── synthetic_gradient/   # SG 的完整源代码
    ├── forward_forward/      # FF 的完整源代码
    └── equilibrium_propagation/ # EP 的完整源代码
```

## 🚀 安装与环境设置

1.  **克隆仓库**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **创建并激活虚拟环境** (推荐)
    ```bash
    # 使用 conda
    conda create -n bp_free python=3.8
    conda activate bp_free

    # 或者使用 venv
    python -m venv venv
    source venv/bin/activate  # 在 Linux/macOS 上
    .\venv\Scripts\activate   # 在 Windows 上
    ```

3.  **安装依赖**
    首先，在项目根目录创建 `requirements.txt` 文件，内容如下：
    ```
    torch
    torchvision
    matplotlib
    hydra-core
    omegaconf
    ```
    然后运行以下命令进行安装：
    ```bash
    pip install -r requirements.txt
    ```

4.  **准备数据集**
    无需手动下载。首次运行针对特定数据集（如CIFAR-10）的训练时，代码会自动从`torchvision`下载并保存到 `./data` 目录。

## 💡 使用方法

所有实验都通过 `main.py` 启动。核心是 `--algorithm` 参数，它决定了要运行哪个算法。

### 基本执行

```bash
python main.py --algorithm <algorithm_name> [COMMON_OPTIONS]
```

**通用选项 (COMMON_OPTIONS):**

- `--algorithm`: **(必需)** 指定要运行的算法。可选值: `sid`, `baseline`, `synthetic_gradient`, `forward_forward`, `equilibrium_propagation`。
- `--dataset`: 使用的数据集。默认为 `cifar10`。可选: `mnist`, `cifar10`。
- `--epochs`: 训练轮数。默认为 `100`。
- `--batch_size`: 批处理大小。默认为 `128`。
- `--learning_rate`: 学习率 (注意: 对于某些算法如EP，此参数可能被其更复杂的学习率策略覆盖)。默认为 `0.01`。
- `--device`: 运行设备。默认为 `cuda` (如果可用)，否则为 `cpu`。

### 指定算法特定参数

每个算法都有其独特的超参数。使用 `--alg_args` 参数，通过一个JSON格式的字符串来传递它们。

**重要提示**: 在命令行中传递JSON字符串时，请注意引号的使用！
-   在 **Linux, macOS, 或 Git Bash** 中，用单引号 `'` 包围整个JSON字符串：
    ```bash
    --alg_args '{"key": "value", "number": 123}'
    ```
-   在 **Windows 命令提示符 (cmd.exe)** 中，用双引号 `"` 包围整个字符串，并用反斜杠 `\` 转义内部的双引号：
    ```cmd
    --alg_args "{\"key\": \"value\", \"number\": 123}"
    ```

---

## 🔬 算法运行示例

以下是为每个算法提供的具体运行命令示例。

### 1. SID 和 Baseline

*(请根据你的 `sid.py` 实现来补充此处的参数和示例)*

```bash
# 运行你的SID算法
python main.py --algorithm sid --dataset cifar10 --epochs 150

# 运行基线方法
python main.py --algorithm baseline --dataset cifar10 --epochs 150
```

### 2. Synthetic Gradients (SG)

**主要可配置参数 (`--alg_args`):**
- `model_type` (str): 模型类型，`cnn` 或 `mlp`。默认为 `cnn`。
- `conditioned` (bool): 是否使用条件DNI。默认为 `false`。

**示例:**
```bash
# 在CIFAR-10上运行CNN模型 (默认配置)
python main.py --algorithm synthetic_gradient --dataset cifar10 --epochs 300 --batch_size 100

# 在MNIST上运行一个条件MLP模型
python main.py --algorithm synthetic_gradient --dataset mnist --epochs 100 \
--alg_args '{"model_type": "mlp", "conditioned": true}'
```

### 3. Forward-Forward (FF)

**主要可配置参数 (`--alg_args`):**
- `model` (dict):
    - `hidden_dim` (int): 隐藏层维度。默认 `1000`。
    - `num_layers` (int): 层数。默认 `3`。
- `training` (dict):
    - `downstream_learning_rate` (float): 分类头学习率。默认 `1e-2`。

**示例:**
```bash
# 使用默认参数在CIFAR-10上运行
python main.py --algorithm forward_forward --dataset cifar10 --epochs 100 --batch_size 100

# 自定义模型层数和隐藏维度
python main.py --algorithm forward_forward --dataset cifar10 --epochs 100 \
--alg_args '{"model": {"num_layers": 4, "hidden_dim": 2048}}'
```

### 4. Equilibrium Propagation (EP)

EP的参数非常多，包装器中已设置了论文中的推荐默认值。

**主要可配置参数 (`--alg_args`):**
- `lrs` (list): 逐层的学习率。
- `betas` (list): EP的beta参数 `[beta_1, beta_2]`。默认 `[0.0, 0.5]`。
- `T1`, `T2` (int): 自由阶段和微扰阶段的迭代步数。默认 `T1=250`, `T2=30`。
- `loss` (str): 损失函数, `mse` 或 `cel`。默认 `mse`。

**示例:**
```bash
# 使用CIFAR-10的推荐默认值运行
python main.py --algorithm equilibrium_propagation --dataset cifar10 --epochs 120 --batch_size 128

# 修改beta值和损失函数为CrossEntropy
python main.py --algorithm equilibrium_propagation --dataset cifar10 --epochs 120 \
--alg_args '{"betas": [0.0, 1.0], "loss": "cel", "softmax": true}'
```

## 📊 查看结果

所有实验的输出，包括日志文件、模型权重 (`.pkl` 或 `.pt`) 和性能图表，都将保存在 `result/` 目录下，并按算法名称分子文件夹存放。

## 🧩 如何添加新算法

本框架的模块化设计使得添加新的BP-Free算法变得简单：

1.  **复制代码**: 将新算法的完整源代码文件夹放入 `algorithms/` 目录下。
2.  **创建`__init__.py`**: 在新算法的根目录中创建一个空的 `__init__.py` 文件，以将其标记为Python包。
3.  **编写包装器**: 在 `algorithms/` 目录下创建一个新的 `new_alg_wrapper.py` 文件。
4.  **实现`run`函数**: 在包装器中，实现一个名为 `run(config)` 的函数。此函数负责：
    -   接收统一的 `config` 字典。
    -   将 `config` 翻译成新算法所需的参数格式（如 `argparse.Namespace`）。
    -   确保算法从 `config['data_path']` 加载数据。
    -   确保算法将结果保存到 `config['result_path']`。
    -   调用新算法的核心训练逻辑。
5.  **整合到主入口**: 在 `main.py` 中，导入你的新包装器，并在主逻辑中添加一个 `elif` 分支来调用它的 `run` 函数。

