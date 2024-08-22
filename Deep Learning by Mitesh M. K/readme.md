# [Deep Learning : Foundation to Advanced](https://study.iitm.ac.in/ds/course_pages/BSCS3004.html)

A brief history of deep learning and its success stories.

Perceptrons, Sigmoid neurons and Multi-Layer Perceptrons (MLP) with specific emphasis on their representation power and algorithms used for training them (such as Perceptron Learning Algorithm and Backpropagation).

Gradient Descent (GD) algorithm and its variants like Momentum based GD,AdaGrad, Adam etc Principal Component Analysis and its relation to modern Autoencoders.

The bias variance tradeoff and regularisation techniques used in DNNs (such as L2 regularisation, noisy data augmentation, dropout, etc).

Different activation functions and weight initialization strategies

Convolutional Neural Networks (CNNs) such as AlexNet, ZFNet, VGGNet, InceptionNet and ResNet.

Recurrent Neural Network (RNNs) and their variants such as LSTMs and GRUs (in particular, understanding the vanishing/exploding gradient problem and how LSTMs overcome the vanishing gradient problem)

Applications of CNN and RNN models for various computer vision and Natural Language Processing (NLP) problems.

## Table of Contents


| WEEK 1  | History of Deep Learning, McCulloch Pitts Neuron, Thresholding Logic, Perceptron Learning Algorithm and Convergence                                               |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WEEK 2  | Multilayer Perceptrons (MLPs), Representation Power of MLPs, Sigmoid Neurons, Gradient Descent                                                                    |
| WEEK 3  | Feedforward Neural Networks, Representation Power of Feedforward Neural Networks, Backpropagation                                                                 |
| WEEK 4  | Gradient Descent(GD), Momentum Based GD, Nesterov Accelerated GD, Stochastic GD, Adagrad, AdaDelta,RMSProp, Adam,AdaMax,NAdam, learning rate schedulers           |
| WEEK 5  | Autoencoders and relation to PCA , Regularization in autoencoders, Denoising autoencoders, Sparse autoencoders, Contractive autoencoders                          |
| WEEK 6  | Bias Variance Tradeoff, L2 regularization, Early stopping, Dataset augmentation, Parameter sharing and tying, Injecting noise at input, Ensemble methods, Dropout |
| WEEK 7  | Greedy Layer Wise Pre-training, Better activation functions, Better weight initialization methods, Batch Normalization                                            |
| WEEK 8  | Learning Vectorial Representations Of Words, Convolutional Neural Networks, LeNet, AlexNet, ZF-Net, VGGNet, GoogLeNet, ResNet                                     |
| WEEK 9  | Visualizing Convolutional Neural Networks, Guided Backpropagation, Deep Dream, Deep Art, Fooling Convolutional Neural Networks                                    |
| WEEK 10 | Recurrent Neural Networks, Backpropagation Through Time (BPTT), Vanishing and Exploding Gradients, Truncated BPTT                                                 |
| WEEK 11 | Gated Recurrent Units (GRUs), Long Short Term Memory (LSTM) Cells, Solving the vanishing gradient problem with LSTM                                               |
| WEEK 12 | Encoder Decoder Models, Attention Mechanism, Attention over images, Hierarchical Attention, Transformers.                                                         |










### Lectures 

Brief history of Deep learning - Neuron Doctrine - AI Winter
[https://www.youtube.com/watch?v=zdmEzqpAG70](https://www.youtube.com/watch?v=zdmEzqpAG70)
The Deep revival - From cats to ConvNet
[https://www.youtube.com/watch?v=EifcfTLIzxc](https://www.youtube.com/watch?v=EifcfTLIzxc)
Faster-Higher-Stronger
[https://www.youtube.com/watch?v=csM_nCAhIQk](https://www.youtube.com/watch?v=csM_nCAhIQk)
Madness  and the rise of transformers
[https://www.youtube.com/watch?v=m2tyQC2qTDQ](https://www.youtube.com/watch?v=m2tyQC2qTDQ)
Call for sanity in AI - Efficient Deep learning
[https://www.youtube.com/watch?v=49F3moGjuwc](https://www.youtube.com/watch?v=49F3moGjuwc)
Motivation from Biological Neuron
[https://www.youtube.com/watch?v=KjMvUwq7PdQ](https://www.youtube.com/watch?v=KjMvUwq7PdQ)
McCulloch Pitts Neuron and Thresholding Logic
[https://www.youtube.com/watch?v=-bxOadOFNYc](https://www.youtube.com/watch?v=-bxOadOFNYc)
Perceptrons
[https://www.youtube.com/watch?v=Ydd9TMyoG6k](https://www.youtube.com/watch?v=Ydd9TMyoG6k)
Errors and error surfaces
[https://www.youtube.com/watch?v=oIjsEzG6zR0](https://www.youtube.com/watch?v=oIjsEzG6zR0)
Perceptron Learning Algorithm
[https://www.youtube.com/watch?v=HPfZ9lEZOj4](https://www.youtube.com/watch?v=HPfZ9lEZOj4)
Proof of Convergence
[https://www.youtube.com/watch?v=9AmGI-tVCSQ](https://www.youtube.com/watch?v=9AmGI-tVCSQ)
Linearly separable functions
[https://www.youtube.com/watch?v=VYnzTwx3lpk](https://www.youtube.com/watch?v=VYnzTwx3lpk)
Representation power of a network of perceptrons
[https://www.youtube.com/watch?v=dFnNoZeuKzQ](https://www.youtube.com/watch?v=dFnNoZeuKzQ)
Sigmoid Neuron
[https://www.youtube.com/watch?v=ay402XiRrzc](https://www.youtube.com/watch?v=ay402XiRrzc)
A typical supervised machine learning setup
[https://www.youtube.com/watch?v=Evhb4HxI32I](https://www.youtube.com/watch?v=Evhb4HxI32I)
Learning Parameters: (Infeasible) guess work
[https://www.youtube.com/watch?v=LbPKf7UVkyU](https://www.youtube.com/watch?v=LbPKf7UVkyU)
Gradient Descent: Weight update rule
[https://www.youtube.com/watch?v=HEx-q6fkCbo](https://www.youtube.com/watch?v=HEx-q6fkCbo)
Learning Parameters: Taylor series approximation
[https://www.youtube.com/watch?v=qyn6RERiIIY](https://www.youtube.com/watch?v=qyn6RERiIIY)
Learning Parameters: Gradient Descent
[https://www.youtube.com/watch?v=o2pT74Nplq0](https://www.youtube.com/watch?v=o2pT74Nplq0)
Representation power of Multilayer Network of Sigmoid Neurons: 1D functions
[https://www.youtube.com/watch?v=xHSErLWAlbU](https://www.youtube.com/watch?v=xHSErLWAlbU)
Representation power of Multilayer Network of Sigmoid Neurons: 2D functions
[https://www.youtube.com/watch?v=BUGi6AH1D2c](https://www.youtube.com/watch?v=BUGi6AH1D2c)
Feed forward neural networks
[https://www.youtube.com/watch?v=HHv6Ndo9VBU](https://www.youtube.com/watch?v=HHv6Ndo9VBU)
Learning parameters (Intuition)
[https://www.youtube.com/watch?v=0Me1ywSlJE8](https://www.youtube.com/watch?v=0Me1ywSlJE8)
Output functions and loss functions
[https://www.youtube.com/watch?v=1hefEWZHvJg](https://www.youtube.com/watch?v=1hefEWZHvJg)
Backpropagation (Intuition)
[https://www.youtube.com/watch?v=i0YC2jZuxUI](https://www.youtube.com/watch?v=i0YC2jZuxUI)
Gradient w.r.t output units
[https://www.youtube.com/watch?v=T6kfyQ3_JQ8](https://www.youtube.com/watch?v=T6kfyQ3_JQ8)
Gradient w.r.t Hidden Units
[https://www.youtube.com/watch?v=7g-THFmBpRA](https://www.youtube.com/watch?v=7g-THFmBpRA)
Gradient w.r.t Parameters
[https://www.youtube.com/watch?v=7TL_9QUIL0g](https://www.youtube.com/watch?v=7TL_9QUIL0g)
BackPropagation: Pseudocode
[https://www.youtube.com/watch?v=1f22ZAS-YGE](https://www.youtube.com/watch?v=1f22ZAS-YGE)
A quick recap on gradient descent and derivative
[https://www.youtube.com/watch?v=gupSH0MU7vs](https://www.youtube.com/watch?v=gupSH0MU7vs)
Plotting contours
[https://www.youtube.com/watch?v=fGzJOBIxXdg](https://www.youtube.com/watch?v=fGzJOBIxXdg)
Momneum based Gradient descent
[https://www.youtube.com/watch?v=R3jlvdclAHI](https://www.youtube.com/watch?v=R3jlvdclAHI)
Nesterov Accelarated Gradient Descent
[https://www.youtube.com/watch?v=dIYDPtHWBNA](https://www.youtube.com/watch?v=dIYDPtHWBNA)
Stochastic vs Batch Gradient
[https://www.youtube.com/watch?v=hXK16fgWjsc](https://www.youtube.com/watch?v=hXK16fgWjsc)
Scheduling learning rate
[https://www.youtube.com/watch?v=-7ET3TUWi8M](https://www.youtube.com/watch?v=-7ET3TUWi8M)
Gradient Descent with Adaptive learning rate
[https://www.youtube.com/watch?v=oqkfhBf71gc](https://www.youtube.com/watch?v=oqkfhBf71gc)
AdaGrad
[https://www.youtube.com/watch?v=WSvxne3oGr0](https://www.youtube.com/watch?v=WSvxne3oGr0)
RMSProp
[https://www.youtube.com/watch?v=ubOy0NPI2cY](https://www.youtube.com/watch?v=ubOy0NPI2cY)
AdaDelta
[https://www.youtube.com/watch?v=EcX0rqjHR9k](https://www.youtube.com/watch?v=EcX0rqjHR9k)
Adam
[https://www.youtube.com/watch?v=m9g9Hij1h1A](https://www.youtube.com/watch?v=m9g9Hij1h1A)
AdaMax and MaxProp
[https://www.youtube.com/watch?v=d2mSof7c4Jo](https://www.youtube.com/watch?v=d2mSof7c4Jo)
NADAM
[https://www.youtube.com/watch?v=fsTRJeWpZ_o](https://www.youtube.com/watch?v=fsTRJeWpZ_o)
Learning rate schemes
[https://www.youtube.com/watch?v=Lqauh9dJzNc](https://www.youtube.com/watch?v=Lqauh9dJzNc)
Introduction to Bias and Variance
[https://www.youtube.com/watch?v=Wrscc4tVmVs](https://www.youtube.com/watch?v=Wrscc4tVmVs)
Training error vs Test error
[https://www.youtube.com/watch?v=RKo-jONCjjU](https://www.youtube.com/watch?v=RKo-jONCjjU)
Estimate error from Test data
[https://www.youtube.com/watch?v=nu6x-L04CG4](https://www.youtube.com/watch?v=nu6x-L04CG4)
True error vs Model complexity
[https://www.youtube.com/watch?v=YNAq1X8b-r4](https://www.youtube.com/watch?v=YNAq1X8b-r4)
L2 Regularization
[https://www.youtube.com/watch?v=eqATcLh_X64](https://www.youtube.com/watch?v=eqATcLh_X64)
Dataset Augmentation and Parameter Sharing
[https://www.youtube.com/watch?v=in9ZdGxGhW8](https://www.youtube.com/watch?v=in9ZdGxGhW8)
Injecting Noise at Inputs
[https://www.youtube.com/watch?v=S1KKaCrg0G8](https://www.youtube.com/watch?v=S1KKaCrg0G8)
Injecting Noise at outputs
[https://www.youtube.com/watch?v=sUnsawJ5-zw](https://www.youtube.com/watch?v=sUnsawJ5-zw)
Early Stopping
[https://www.youtube.com/watch?v=CrO__rzk9hU](https://www.youtube.com/watch?v=CrO__rzk9hU)
Ensemble Methods
[https://www.youtube.com/watch?v=Njytanc4oHY](https://www.youtube.com/watch?v=Njytanc4oHY)
Dropout
[https://www.youtube.com/watch?v=WzScUPDGFVA](https://www.youtube.com/watch?v=WzScUPDGFVA)
Summary
[https://www.youtube.com/watch?v=_wK-MOylRtw](https://www.youtube.com/watch?v=_wK-MOylRtw)
Deep Learning revival
[https://www.youtube.com/watch?v=PwKAEs9uuZw](https://www.youtube.com/watch?v=PwKAEs9uuZw)
Unsupervised Pre-Training
[https://www.youtube.com/watch?v=TNcZ5l54pLc](https://www.youtube.com/watch?v=TNcZ5l54pLc)
Optimization or Regularization?
[https://www.youtube.com/watch?v=n5ZP4Ek9cxE](https://www.youtube.com/watch?v=n5ZP4Ek9cxE)
Better Activation functions
[https://www.youtube.com/watch?v=4eQiXN69FdQ](https://www.youtube.com/watch?v=4eQiXN69FdQ)
MaxOut
[https://www.youtube.com/watch?v=oW7ZBLss2L0](https://www.youtube.com/watch?v=oW7ZBLss2L0)
GELU to SILU
[https://www.youtube.com/watch?v=srcPpAV6Yfs](https://www.youtube.com/watch?v=srcPpAV6Yfs)
Convolutional neural networks
[https://www.youtube.com/watch?v=Dsq-fSb_lj4](https://www.youtube.com/watch?v=Dsq-fSb_lj4)
Introduction to transformer architecture
[https://www.youtube.com/watch?v=cVbGNL0N2RI](https://www.youtube.com/watch?v=cVbGNL0N2RI)
Attention is all you need
[https://www.youtube.com/watch?v=MVeOwsggkt4](https://www.youtube.com/watch?v=MVeOwsggkt4)
Self-Attention
[https://www.youtube.com/watch?v=MLdUzA6ltEQ](https://www.youtube.com/watch?v=MLdUzA6ltEQ)
Multi-headed attention
[https://www.youtube.com/watch?v=VN2Bfi0_pbw](https://www.youtube.com/watch?v=VN2Bfi0_pbw)
Decoder block
[https://www.youtube.com/watch?v=cVuyWE4yMBU](https://www.youtube.com/watch?v=cVuyWE4yMBU)
"Analysis of K-Maps In Complex Boolean LogicMinimization"
[https://www.youtube.com/watch?v=_yFMy5WrKSk](https://www.youtube.com/watch?v=_yFMy5WrKSk)


## Books and References


Ian Goodfellow and Yoshua Bengio and Aaron Courville. Deep Learning. An MIT Press book. 2016.

Charu C. Aggarwal. Neural Networks and Deep Learning: A Textbook. Springer. 2019.

## Summary
