# algorithms/ep_wrapper.py

import argparse
import torch
import torchvision
import os
import sys
    
from .equilibrium_propagation.model_utils import *
from .equilibrium_propagation.data_utils import *



def run_ep(config):
    """
    Wrapper for the Equilibrium-Propagation algorithm.
    This function translates the unified config into the format expected by the original EP script,
    and then executes its training logic.
    """
    # --------------------------------------------------------------------------
    # Step 1: Translate the standard config into the argparse.Namespace object
    # that the original Equilibrium-Propagation script expects.
    # --------------------------------------------------------------------------
    args = argparse.Namespace()
    
    # ---- Map common parameters from the unified config ----
    args.task = 'CIFAR10' if config['dataset'] == 'cifar10' else 'MNIST'
    args.epochs = config['epochs']
    args.mbs = config['batch_size']
    # EP code expects an integer for the CUDA device index
    if 'cuda' in config['device']:
        try:
            args.device = int(config['device'].split(':')[-1])
        except (ValueError, IndexError):
            args.device = 0 # Default to cuda:0 if parsing fails
    else:
        args.device = 'cpu'


    # ---- Set algorithm-specific parameters with reasonable defaults ----
    # These defaults are based on the CIFAR10 CNN example in the original README.md.
    # They can be overridden by the 'alg_specific_args' in the main config.
    alg_args = config.get('alg_specific_args', {})
    args.model = alg_args.get('model', 'CNN')
    args.channels = alg_args.get('channels', [128, 256, 512, 512])
    args.kernels = alg_args.get('kernels', [3, 3, 3, 3])
    args.pools = alg_args.get('pools', 'mmmm')
    args.strides = alg_args.get('strides', [1, 1, 1, 1])
    args.paddings = alg_args.get('paddings', [1, 1, 1, 0])
    args.fc = alg_args.get('fc', [10])
    
    args.optim = alg_args.get('optim', 'sgd')
    # Learning rate from main config is ignored here, as EP uses layer-wise rates.
    args.lrs = alg_args.get('lrs', [0.25, 0.15, 0.1, 0.08, 0.05])
    args.wds = alg_args.get('wds', [3e-4, 3e-4, 3e-4, 3e-4, 3e-4])
    args.mmt = alg_args.get('mmt', 0.9)
    args.lr_decay = alg_args.get('lr_decay', True)
    
    args.act = alg_args.get('act', 'my_hard_sig')
    args.loss = alg_args.get('loss', 'mse')
    args.alg = alg_args.get('alg', 'EP')
    
    args.T1 = alg_args.get('T1', 250)
    args.T2 = alg_args.get('T2', 30)
    args.betas = alg_args.get('betas', [0.0, 0.5])
    
    args.data_aug = alg_args.get('data_aug', True if args.task == 'CIFAR10' else False)
    args.thirdphase = alg_args.get('thirdphase', True)
    args.softmax = alg_args.get('softmax', False if args.loss == 'mse' else True)
    args.seed = alg_args.get('seed', 42)

    # Hardcode parameters that are fixed for this unified runner
    args.save = True
    args.load_path = ''
    args.todo = 'train'
    args.random_sign = False
    args.check_thm = False
    args.scale = None
    args.cep_debug = False
    args.same_update = False


    # --------------------------------------------------------------------------
    # Step 2: Replicate the setup logic from the original main.py, but using
    # our unified data and result paths.
    # --------------------------------------------------------------------------
    
    print("\n--- Effective Equilibrium-Propagation Config ---")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("----------------------------------------------\n")

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and isinstance(args.device, int) else 'cpu')
    
    # CRITICAL: Use the result path from the main config
    path = config['result_path'] 
    
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ---- Data Loading ----
    # CRITICAL: Use the data path from the main config
    data_root = config['data_path']

    if args.task == 'MNIST':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])
        train_dset = torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
        test_dset = torchvision.datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.mbs, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dset, batch_size=200, shuffle=False, num_workers=2)
    
    elif args.task == 'CIFAR10':
        if args.data_aug:
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3*0.2023, 3*0.1994, 3*0.2010))
            ])   
        else:
             transform_train = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3*0.2023, 3*0.1994, 3*0.2010))
            ])   
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3*0.2023, 3*0.1994, 3*0.2010))
        ]) 
        cifar10_train_dset = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True)
        cifar10_test_dset = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=transform_test, download=True)
        train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=args.mbs, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=200, shuffle=False, num_workers=2)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # ---- Activation and Loss Function Selection (copied from original main.py) ----
    activations = {
        'mysig': my_sigmoid, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh,
        'hard_sigmoid': hard_sigmoid, 'my_hard_sig': my_hard_sig, 'ctrd_hard_sig': ctrd_hard_sig
    }
    activation = activations.get(args.act, my_sigmoid)
    
    criterion = torch.nn.MSELoss(reduction='none').to(device) if args.loss == 'mse' else torch.nn.CrossEntropyLoss(reduction='none').to(device)
    print('loss =', criterion, '\n')

    # ---- Model Creation (copied from original main.py) ----
    if args.model == 'MLP':
        model = P_MLP(args.archi, activation=activation)
    elif args.model.find('CNN') != -1:
        pools = make_pools(args.pools)
        if args.task == 'MNIST':
            channels = [1] + args.channels
            input_size = 28
        elif args.task == 'CIFAR10':
            channels = [3] + args.channels
            input_size = 32
        else:
            raise ValueError("Task must be MNIST or CIFAR10 for CNNs in this wrapper.")

        if args.model == 'CNN':
            model = P_CNN(input_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                          activation=activation, softmax=args.softmax)
        elif args.model == 'VFCNN':
            model = VF_CNN(input_size, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                           activation=activation, softmax=args.softmax, same_update=args.same_update)
        print('\nPoolings =', model.pools)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    if args.scale is not None:
        model.apply(my_init(args.scale))
        
    model.to(device)
    print(model)

    betas = args.betas[0], args.betas[1]

    # ---- Optimizer and Scheduler Setup (copied from original main.py) ----
    optim_params = []
    for idx in range(len(model.synapses)):
        param_group = {'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx]}
        if args.wds is not None:
            param_group['weight_decay'] = args.wds[idx]
        optim_params.append(param_group)

    if hasattr(model, 'B_syn'):
        for idx in range(len(model.B_syn)):
            param_group = {'params': model.B_syn[idx].parameters(), 'lr': args.lrs[idx+1]}
            if args.wds is not None:
                 param_group['weight_decay'] = args.wds[idx+1]
            optim_params.append(param_group)
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(optim_params, momentum=args.mmt)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(optim_params)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optim}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5) if args.lr_decay else None
    
    print(optimizer)


    # --------------------------------------------------------------------------
    # Step 3: Call the core training function from model_utils.py.
    # --------------------------------------------------------------------------
    print(f'\nTraining algorithm: {args.alg}\n')
    if args.save:
        # We can save our own config file instead of using their helper
        with open(os.path.join(path, 'hyperparameters.txt'), 'w') as f:
            f.write("--- Unified Runner Config ---\n")
            for k, v in config.items():
                f.write(f"{k}: {v}\n")
            f.write("\n--- Effective EP Args ---\n")
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    train(model, optimizer, train_loader, test_loader, args.T1, args.T2, betas, device, args.epochs, criterion, 
          alg=args.alg, random_sign=args.random_sign, check_thm=args.check_thm, save=args.save, path=path, 
          checkpoint=None, thirdphase=args.thirdphase, scheduler=scheduler, cep_debug=args.cep_debug)