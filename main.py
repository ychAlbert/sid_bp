# main.py
import argparse
import os
import torch
import json

# 导入各个算法的包装器
from algorithms.sg_wrapper import run_sg
from algorithms.ff_wrapper import run_ff
from algorithms.ep_wrapper import run_ep

# from sid import run_sid 

def main():
    parser = argparse.ArgumentParser(description="Unified Runner for BP-Free Algorithms")
    parser.add_argument('--algorithm', type=str, required=True, 
                        choices=['sid', 'baseline', 'synthetic_gradient', 'forward_forward', 'equilibrium_propagation'],
                        help='Algorithm to run.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'], help='Dataset to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    
    # 使用一个灵活的参数来传递特定于算法的配置
    parser.add_argument('--alg_args', type=str, default='{}', help='JSON string of algorithm-specific arguments.')

    args = parser.parse_args()

    # ---- 1. 构建标准化的配置字典 ----
    # 确保 data 和 result 目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    result_path = os.path.join('result', args.algorithm)
    os.makedirs(result_path, exist_ok=True)

    config = {
        'algorithm': args.algorithm,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'data_path': os.path.abspath('./data'),
        'result_path': os.path.abspath(result_path),
        'alg_specific_args': json.loads(args.alg_args) # 解析JSON字符串为字典
    }
    
    print("--- Running Experiment with Configuration ---")
    print(json.dumps(config, indent=2))
    print("-------------------------------------------")

    # ---- 2. 根据算法名称调用对应的包装器 ----
    if args.algorithm == 'synthetic_gradient':
        run_sg(config)
    elif args.algorithm == 'forward_forward':
        run_ff(config)
    elif args.algorithm == 'equilibrium_propagation':
        run_ep(config)
    # elif args.algorithm == 'sid':
    #     run_sid(config)
    # elif args.algorithm == 'baseline':
    #     run_baseline(config)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    print(f"--- Experiment for {args.algorithm} finished. Results saved in {config['result_path']} ---")


if __name__ == '__main__':
    main()