# algorithms/ff_wrapper.py
import time
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig
import torch
# 确保你的环境安装了 omegaconf: pip install omegaconf
# from .forward_forward import main as ff_main # 避免直接导入，因为有@hydra.main
from .forward_forward.src import utils

def run_ff(config):
    """
    Wrapper for the Forward-Forward algorithm.
    """
    # 1. 加载默认的yaml配置文件
    # 路径是相对于当前工作目录，所以要确保正确
    default_conf_path = 'algorithms/forward_forward/config.yaml'
    opt = OmegaConf.load(default_conf_path)

    # 2. 使用我们的 config 字典来覆盖/合并配置
    # ---- 通用配置 ----
    opt.device = config['device']
    opt.input.path = config['data_path'] # <-- 关键：设置数据路径
    opt.input.batch_size = config['batch_size']
    opt.training.epochs = config['epochs']
    opt.training.learning_rate = config['learning_rate']
    
    # ---- 结果保存路径 ----
    # Hydra默认会创建自己的目录结构，我们把它重定向到我们的 result 目录
    opt.hydra.run.dir = config['result_path'] 

    # ---- 算法特定配置 ----
    alg_args = config.get('alg_specific_args', {})
    if 'model' in alg_args:
        opt.model = OmegaConf.merge(opt.model, alg_args['model'])
    if 'training' in alg_args:
        opt.training = OmegaConf.merge(opt.training, alg_args['training'])
    
    # 确保数据集名称在配置中
    # FF代码似乎从opt.input.dataset推断数据集，但配置文件中没有，我们加上
    opt.input.dataset = 'CIFAR10' if config['dataset'] == 'cifar10' else 'MNIST'


    print("--- Effective Forward-Forward Config ---")
    print(OmegaConf.to_yaml(opt))
    print("----------------------------------------")

    # 3. 复用原始代码的逻辑，但不通过 @hydra.main 启动
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    
    # 下面的代码直接从 forward_forward/main.py 的 train 和 validate_or_test 函数复制而来
    # 这样做可以避免处理Hydra的上下文装饰器带来的麻烦
    
    # --- Start Training Logic (copied from original main.py) ---
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)
            optimizer.zero_grad()
            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()
            optimizer.step()
            train_results = utils.log_results(train_results, scalar_outputs, num_steps_per_epoch)

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", epoch=epoch)
    
    # --- Final Validation ---
    validate_or_test(opt, model, "val")
    if opt.training.final_test:
        validate_or_test(opt, model, "test")


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)
    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)
            scalar_outputs = model.forward_downstream_classification_model(inputs, labels)
            test_results = utils.log_results(test_results, scalar_outputs, num_steps_per_epoch)
    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()