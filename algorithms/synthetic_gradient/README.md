**disclaimer**: this code is modified from [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

# Image classification with synthetic gradient in Pytorch
I implement the ***[Decoupled Neural Interfaces using Synthetic Gradients](http://arxiv.org/abs/1608.05343)*** in **pytorch**. The paper uses synthetic gradient to decouple the layers among the network, which is pretty interesting since we won't suffer from **update lock** anymore. I test my model in mnist and **almost** the same performance, compared to the model updated with backpropagation.

## Requirement
- pytorch
- python 3.5
- torchvision
- seaborn (optional)
- matplotlib (optional)

## TODO
- use multi-threading on gpu to analyze the speed

## What's synthetic gradients?
We ofter optimize NN by backpropogation, which is usually implemented in some well-known framework. However, is there another way for the layers in NN to communicate with other layers? Here comes the ***synthetic gradients***! It gives us a way to allow neural networks to communicate, to learn to send messages between themselves, in a decoupled, scalable manner paving the way for multiple neural networks to communicate with each other or improving the long term temporal dependency of recurrent networks.   
The neuron in each layer will automatically produces an error signal(***δa_head***) from synthetic-layers and do the optimzation. And how did the error signal generated? Actually, the network still does the backpropogation. While the error signal(***δa***) from the objective function is not used to optimize the neuron in the network, it is used to optimize the error signal(***δa_head***) produced by the synthetic-layer. The following is the illustration from the paper:
## 什么是合成梯度？
我们通常通过反向传播来优化神经网络，这通常在一些知名框架中实现。然而，神经网络中的各层之间是否还有其他通信方式呢？这时就出现了***合成梯度***！它为我们提供了一种方式，使神经网络能够以解耦、可扩展的方式进行通信，学会在彼此之间传递信息，为多个神经网络相互通信或改善循环网络的长期时间依赖关系铺平了道路。  
每一层的神经元将自动从合成层产生一个误差信号（***δa_head***）并进行优化。那么这个误差信号是如何产生的呢？实际上，网络仍然要进行反向传播。不过，来自目标函数的误差信号（***δa***）并非用于优化网络中的神经元，而是用于优化由合成层产生的误差信号（***δa_head***）。以下是论文中的图示： 
![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/dni_illustration.png?raw=true)   

## Result

### Feed-Forward Network

Achieve accuracy=**96%** (compared to the original model, which with accuracy=97%)

| classify loss | gradient loss(log level) |
|----|----|
| ![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/classify_loss.png?raw=true) | ![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/grad_loss.png?raw=true) |

| cDNI classify loss | cDNI gradient loss(log level) |
|----|----|
| ![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/cDNI_classify_loss.png?raw=true) | ![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/cDNI_grad_loss.png?raw=true) |

### Convolutional Neural Network

Achieve accuracy=**96%**, (compared to the original model, which with accuracy=98%)

| classify loss | gradient loss(log level) |
|----|----|
| ![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/cnn_classify_loss.png?raw=true) | ![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/cnn_grad_loss.png?raw=true) |

## Usage 
Right now I just implement the FCN, CNN versions, which are set as the default network structure.

### Run network with synthetic gradient:

```
python main.py --model_type mlp
```

or 

```
python main.py --model_type cnn
```

### Run network with conditioned synthetic gradient:

```
python main.py --model_type mlp --conditioned True
```

### Run vanilla network, from pytorch-tutorial
```python
python mlp.py
```

or 

```python
python cnn.py
```

## Reference
- Deepmind's [post](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/) on Decoupled Neural Interfaces Using Synthetic Gradients
- [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343)
- [Understanding Synthetic Gradients and Decoupled Neural Interfaces](https://arxiv.org/abs/1703.00522)

