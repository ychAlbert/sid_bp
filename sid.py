import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
import os

# ==============================================================================
# 0. GPU Memory Utility (新增部分)
# ==============================================================================
def get_gpu_memory_usage(device):
    """Returns the current and peak GPU memory usage in MB for a given device."""
    if device.type != 'cuda':
        return 0, 0
    # torch.cuda.memory_allocated()：返回当前张量占用的内存（以字节为单位）
    # torch.cuda.max_memory_allocated()：返回此进程在GPU上分配的最大内存
    # 我们用 reset_peak_memory_stats 来确保峰值是每个 epoch 的峰值
    current_mem = torch.cuda.memory_allocated(device) / 1024**2
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    return current_mem, peak_mem

def reset_gpu_memory_stats(device):
    """Resets the peak memory statistics for a given CUDA device."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
# ==============================================================================

# ==============================================================================
# 1. Configuration via Argparse
# ==============================================================================
def get_config():
    parser = argparse.ArgumentParser(description="SID vs Backprop vs NoProp Comparative Experiments")
    parser.add_argument('--method', type=str, default='sid', choices=['sid', 'backprop', 'noprop'],
                        help='Training method to use')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'miniimagenet'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_modules', type=int, default=10, help='Number of modules/steps (depth)')
    parser.add_argument('--alpha', type=float, default=0.5, help='SID loss balance parameter')
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True)

    try:
        # For command line execution
        args = parser.parse_args()
    except (SystemExit, TypeError):
        # Default args for interactive environments (e.g., Jupyter, VSCode Notebook)
        # Manually set args for testing. Change '--dataset' to 'cifar10' or 'miniimagenet' to test others.
        args = parser.parse_args(args=['--dataset', 'mnist', '--method', 'sid'])
        
    return args

# ==============================================================================
# 2. Data Loading Module
# ==============================================================================
def get_dataloaders(dataset_name, batch_size):
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        num_classes, in_channels = 10, 1
    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('data', train=False, transform=transform_test)
        num_classes, in_channels = 10, 3
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        train_dataset = datasets.CIFAR100('data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
        num_classes, in_channels = 100, 3
    elif dataset_name == 'miniimagenet':
        data_path = './mini_imagenet100'
        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"Mini-ImageNet directory not found at '{data_path}'.\n"
                "Please clone the dataset first by running:\n"
                "git clone https://www.modelscope.cn/datasets/tany0699/mini_imagenet100.git"
            )
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(84), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(92), transforms.CenterCrop(84), transforms.ToTensor(), normalize,
        ])
        train_path = os.path.join(data_path, 'train')
        test_path = os.path.join(data_path, 'val')
        train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
        test_dataset = datasets.ImageFolder(root=test_path, transform=transform_test)
        num_classes, in_channels = 100, 3
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, num_classes, in_channels

# ==============================================================================
# 3. Model Architectures (No changes here)
# ==============================================================================
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, feature_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )
    def forward(self, x):
        return self.cnn(x)

class ProcessingModule(nn.Module):
    def __init__(self, num_classes, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, p_prev, shared_features):
        combined_input = torch.cat((p_prev, shared_features), dim=1)
        return self.mlp(combined_input)

class NoProp_Network(nn.Module):
    def __init__(self, num_steps, num_classes, in_channels, feature_dim=128):
        super().__init__()
        self.T = num_steps
        self.num_classes = num_classes
        self.cnn = FeatureExtractor(in_channels, feature_dim)
        self.mlps = nn.ModuleList([ProcessingModule(num_classes, feature_dim) for _ in range(self.T)])
        self.register_buffer('alpha', torch.linspace(1.0, 0.1, self.T))
    def forward(self, x):
        batch_size = x.shape[0]
        z_t = torch.randn(batch_size, self.num_classes, device=x.device)
        x_features = self.cnn(x)
        for t in reversed(range(self.T)):
            z_t = self.mlps[t](z_t, x_features)
        return z_t

class SID_Network(nn.Module):
    def __init__(self, num_modules, num_classes, in_channels, feature_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = FeatureExtractor(in_channels, feature_dim)
        self.processing_modules = nn.ModuleList([ProcessingModule(num_classes, feature_dim) for _ in range(num_modules)])
    def forward(self, x):
        batch_size = x.shape[0]
        p_current = torch.full((batch_size, self.num_classes), 1.0 / self.num_classes, device=x.device)
        shared_features = self.feature_extractor(x)
        for module in self.processing_modules:
            logits = module(p_current, shared_features)
            p_current = F.softmax(logits, dim=1)
        return logits

class Backprop_Network(nn.Module):
    def __init__(self, num_modules, num_classes, in_channels, feature_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = FeatureExtractor(in_channels, feature_dim)
        self.processing_modules = nn.ModuleList([ProcessingModule(num_classes, feature_dim) for _ in range(num_modules)])
    def forward(self, x):
        batch_size = x.shape[0]
        p_current = torch.full((batch_size, self.num_classes), 1.0 / self.num_classes, device=x.device)
        shared_features = self.feature_extractor(x)
        for module in self.processing_modules:
            logits = module(p_current, shared_features)
            p_current = F.softmax(logits, dim=1)
        return logits

# ==============================================================================
# 4. Training and Evaluation Loops (修改部分)
# ==============================================================================

# --- Helper function to update tqdm with memory usage ---
def update_tqdm_postfix(pbar, loss, device):
    """Updates tqdm postfix with loss and current GPU memory."""
    if device.type == 'cuda':
        mem_current, _ = get_gpu_memory_usage(device)
        pbar.set_postfix(loss=loss, mem=f"{mem_current:.1f}MB")
    else:
        pbar.set_postfix(loss=loss)

# --- Training functions updated to use the helper ---
def train_noprop(model, train_loader, optimizers, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="NoProp Training", leave=False)
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        u_y = F.one_hot(y, num_classes=model.num_classes).float()
        
        z_t_plus_1_list = []
        for t in range(model.T):
            eps = torch.randn_like(u_y)
            z_t_plus_1 = torch.sqrt(model.alpha[t]) * u_y + torch.sqrt(1 - model.alpha[t]) * eps
            z_t_plus_1_list.append(z_t_plus_1)
            
        x_features = model.cnn(x)
        losses = []
        for t in range(model.T):
            u_hat = model.mlps[t](z_t_plus_1_list[t].detach(), x_features)
            losses.append(F.mse_loss(u_hat, u_y))
            
        total_loss_batch = sum(losses)
        for opt in optimizers: opt.zero_grad()
        total_loss_batch.backward()
        for opt in optimizers: opt.step()
        
        total_loss += total_loss_batch.item()
        update_tqdm_postfix(progress_bar, total_loss_batch.item(), device)
        
    return total_loss / len(train_loader)

def train_sid_simplified(model, train_loader, optimizers, num_classes, device, alpha):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"SID Training", leave=False)
    
    opt_cnn = optimizers[0]
    opt_mlps = optimizers[1:]

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        y_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        
        opt_cnn.zero_grad()
        for opt in opt_mlps: opt.zero_grad()

        shared_features = model.feature_extractor(images)
        p_inputs = []
        with torch.no_grad():
            p_current = torch.full((images.size(0), num_classes), 1.0/num_classes, device=device)
            p_inputs.append(p_current)
            detached_features = shared_features.detach()
            for module in model.processing_modules:
                logits = module(p_current, detached_features)
                p_current = F.softmax(logits, dim=1)
                p_inputs.append(p_current)

        final_loss = 0
        for i, module in enumerate(model.processing_modules):
            p_prev = p_inputs[i].detach()
            logits_i = module(p_prev, shared_features)
            log_p_i = F.log_softmax(logits_i, dim=1)
            loss_target = F.kl_div(log_p_i, y_one_hot, reduction='batchmean')
            loss_consistency = F.kl_div(log_p_i, p_prev, reduction='batchmean')
            final_loss += (alpha * loss_target + (1 - alpha) * loss_consistency)

        final_loss.backward()
        opt_cnn.step()
        for opt in opt_mlps: opt.step()
        
        total_loss += final_loss.item()
        update_tqdm_postfix(progress_bar, final_loss.item(), device)
            
    return total_loss / len(train_loader)

def train_backprop(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Backprop Training", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        update_tqdm_postfix(progress_bar, loss.item(), device)
        
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ==============================================================================
# 5. Main Execution Block (修改部分)
# ==============================================================================
if __name__ == '__main__':
    args = get_config()
    DEVICE = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    print("-" * 50)
    print(f"Starting experiment with config:")
    print(f"  Method: {args.method.upper()}")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"  Network Depth (L or T): {args.num_modules}")
    if args.method == 'sid': print(f"  SID Alpha: {args.alpha}")
    print(f"  Device: {DEVICE}")
    print("-" * 50)

    train_loader, test_loader, num_classes, in_channels = get_dataloaders(args.dataset, args.batch_size)
    
    if args.method == 'sid':
        model = SID_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
        optimizers = [optim.Adam(model.feature_extractor.parameters(), lr=args.lr)]
        optimizers.extend([optim.Adam(m.parameters(), lr=args.lr) for m in model.processing_modules])
        train_fn = train_sid_simplified
        train_args = {'optimizers': optimizers, 'num_classes': num_classes, 'device': DEVICE, 'alpha': args.alpha}
    
    elif args.method == 'noprop':
        model = NoProp_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
        optimizers = [optim.Adam(model.cnn.parameters(), lr=args.lr)]
        optimizers.extend([optim.Adam(mlp.parameters(), lr=args.lr) for mlp in model.mlps])
        train_fn = train_noprop
        train_args = {'optimizers': optimizers, 'device': DEVICE}

    else: # backprop
        model = Backprop_Network(args.num_modules, num_classes, in_channels).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        train_fn = train_backprop
        train_args = {'optimizer': optimizer, 'criterion': criterion, 'device': DEVICE}

    best_accuracy = 0.0
    total_training_time = 0
    
    # 记录训练前的初始显存占用（主要为模型参数和CUDA上下文）
    initial_mem_current, _ = get_gpu_memory_usage(DEVICE)
    print(f"Initial GPU memory usage: {initial_mem_current:.2f} MB")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        # 在每个 epoch 开始前，重置峰值显存统计
        reset_gpu_memory_stats(DEVICE)
        
        # 训练
        avg_loss = train_fn(model, train_loader, **train_args)
        
        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        
        # 评估
        accuracy = evaluate(model, test_loader, DEVICE)
        
        # 获取该 epoch 的峰值显存
        _, peak_mem_epoch = get_gpu_memory_usage(DEVICE)
        
        # 修改打印输出，加入显存信息
        print(f"  Avg Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}% | Time: {epoch_time:.2f}s | Peak Mem: {peak_mem_epoch:.2f}MB")
        
        if accuracy > best_accuracy: best_accuracy = accuracy
            
    print("\n" + "=" * 50)
    print("Experiment Finished!")
    print(f"Method: {args.method.upper()}, Dataset: {args.dataset.upper()}")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Average Time per Epoch: {total_training_time/args.epochs:.2f}s")
    print("=" * 50)