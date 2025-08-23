# algorithms/sg_wrapper.py
import argparse
import torch
from .synthetic_gradient.train import classifier
from .synthetic_gradient.dataset import mnist, cifar10
import os
import sys
ep_path = os.path.abspath('algorithms/synthetic_gradient')
if ep_path not in sys.path:
    sys.path.append(ep_path)
def run_sg(config):
    """
    Wrapper for the synthetic-gradient algorithm.
    """
    # 1. 将我们的 config 字典翻译成 synthetic-gradient 所需的 argparse.Namespace 对象
    args = argparse.Namespace()
    
    # 从 config 映射通用参数
    args.dataset = config['dataset']
    args.num_epochs = config['epochs']
    args.batch_size = config['batch_size']
    args.use_gpu = (config['device'] == 'cuda')

    # 设置算法特定参数的默认值，并允许通过 config['alg_specific_args'] 覆盖
    alg_args = config.get('alg_specific_args', {})
    args.model_type = alg_args.get('model_type', 'cnn')
    args.conditioned = alg_args.get('conditioned', False)
    args.plot = alg_args.get('plot', False)

    # 检查逻辑
    if args.dataset == 'cifar10' and args.model_type == 'mlp':
        print("Warning: synthetic_gradient does not support MLP for CIFAR10. Skipping.")
        return

    # 添加结果路径到 args 中，供 train.py 使用
    args.result_path = config['result_path']

    # 2. 准备模型名称和数据集
    model_name = f"{args.dataset}.{args.model_type}_dni"
    if args.conditioned:
        model_name += '.conditioned'
    args.model_name = model_name

    print(f"Running Synthetic Gradient with model: {model_name}")

    if args.dataset == 'mnist':
        data = mnist(args)
    elif args.dataset == 'cifar10':
        data = cifar10(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 3. 实例化并运行训练
    m = classifier(args, data)
    m.train_model()