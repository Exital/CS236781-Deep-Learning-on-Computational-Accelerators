r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1.  The Jacobian's tensor shape of the output of the layer w.r.t the input X is of shape (batch, ins, batch, outs).\n
    Therefore it's (N, in_features, N, out_features) = (128,1024,128,2048).
    
    

2.  In order to store the Jacobian in memory of GPU or RAM we will need 128GB.
    Assuming we are using single-precision floating point (32 bits) to represent our tensors:
    we have 128 batches with 128 samples each, and in_features = 1024, out_features = 2048.
    The computation is:
    $(num of samples) \cdot  (num of batches) \cdot (features_{in}) \cdot (features_{out}) \cdot 4(bytes)
    = 4 \cdot 128 \cdot 128 \cdot 1024 \cdot 2048 = 2^2 \cdot 2^7 \cdot 2^7 \cdot 2^{10} \cdot 2^{11} = 2^{37}
    = 2^7 \cdot 2^{30} = 128 \cdot 2^{30} = ~ 128GB$
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.01, 0.08, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr_vanilla = 0.01
    lr_momentum = 0.001
    lr_rmsprop = 0.0001
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.003
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1.  Dropout is a regularization technique for reducing overfitting.
    Therefore we are expecting to see better test results with Dropout.
    For example lets compare the results from our Graph between drop=0 (no-dropout) and drop=0.4(optimal dropout).
    On the training set, no-dropout is doing much better but it's overfitiing to those samples, that's why it hearts
    the performance on more generalized samples.
    On the other hand we can see that dropout=0.4 is doing much worse on the training set but doing much better on the tests.
    This is exactly what we excepted of dropout.
    
2.  As we can see from the graph the lower dropout setting is doing much better on tests and training then the
    higher setting. On high setting we lose a lot of data thus creating much less data to work with and leading to
    lower results.
"""

part2_q2 = r"""
**Your answer:**
Although it seems that accuracy and loss are inversely correlated there are cases which
it is possible for the test loss to increase for a few epochs while the test accuracy also increases.

Loss measures a difference between **raw prediction** and class.
Accuracy measure a difference between **threshold prediction** and class.

Therefore if **raw predictions** changes - loss is effected right away where accuracy is more "resilient"
as the predictions has to go over or under a threshold in order to change.

Also the loss function is continuous while accuracy is correct or not correct.
Therefore there might be some epochs where a prediction has increased loss but still predict correctly.

So if a few raw predictions were bad but the model still predicts right it causes the loss to increase, while accuracy stays the same.
At the same time the model is still learning patterns which are useful for generalization therefore increases accuracy.  


"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1.  Regular block has 2 convolution layers, each has 3X3 kernel and 256 channels as input and output therefore it's parameters count is: $2\cdot((3\cdot3\cdot256 + 1)\cdot256) = 1,180,160$.
    
    Bottleneck block has 3 convolution layers:
        a. 1X1 256 -> 64
        b. 3X3 64 -> 64
        c. 1X1 64 -> 256
    Therefore it's parameters count is $(256+1)\cdot64 + (3\cdot3\cdot64 + 1)\cdot64 + (64+1)\cdot256 = 70,016$.
    
    We can see that the bottleneck has less parameters than the regular block thus it needs much less resources as memory and computing power in order to computre.


2.  Lets assume input size is (256, H, W) and that the following are one floating point operation:
        a. basic math operations
        b. Relu (per element)
    The formula is $K \cdot K \cdot input \cdot H_{out} \cdot W_{out} \cdot output$ for a KxK input -> output layer forward pass.
    
    The regular block:
    
    First layer = $3 \cdot 3 \cdot 256 \cdot H \cdot W \cdot 256$
    
    2 X Relu =  $2 \cdot 256 \cdot H \cdot W $
    
    Second layer = $ 3 \cdot 3 \cdot 256 \cdot H \cdot W \cdot 256 $
    
    Residual connection = $ 256 \cdot H \cdot W $
    
    All of those sum up to $ 1,180,416 \cdot H \cdot W$
    
    The bottleneck block:
    
    First layer = $1 \cdot 1 \cdot 256 \cdot H \cdot W \cdot 64 $
    
    2 X Relu (64) = $2 \cdot  64 \cdot H \cdot W $
    
    Second layer = $3 \cdot 3 \cdot 64 \cdot H \cdot W \cdot 64 $ 
    
    Third layer = $1 \cdot 1 \cdot 64 \cdot H \cdot W \cdot 256 $
    
    Residual connection = $256 \cdot H \cdot W $
    
    Relu (256) = $256 \cdot H \cdot W $
    
    All of those sum up to $70,272 \cdot H \cdot W$
    
    In conclusion from those calculations we can see that the bottleneck block is much better in terms of computing.


3.  The regular block mostly effects the input spatially - we can conclude it because it maintains the number of feature maps.

    The bottleneck block effects both spatial and feature map of the input - we can conclude it because the first convolution layer 1X1 maintaining the spatiallity of the input
    and map the input, afterwards the input goes through a similar convolutions as the regular block that operates on the spatiallity of the input.
    Therefore we conclude that the bottleneck block effects mostly on the spatial aspect but also on the feature map.
"""

part3_q2 = r"""
Our conclusions from the graph:

1.  We can see that the best results are with $L=2,4$ and as we go higher with **L** the network becomes untrainable as we can see with $L=8,16$.
    We have read on the internet and encountered a phenomena that is called **Vanishing gradient problem**.
    It occurs in deeper networks and causes the network to perform bad or untrainable.
    
    Solutions:
    
    a. Our Resnet should help with that case as it has shortcuts that allow skipping layers and that way solves the depth problem.
    
    b. We have read online that Long Short-Term Memory Networks (LSTMs), which were pioneered by Sepp Hochreiter (the guy who found out of Vanishing gradient problem) are a good solution to that problem.
    
    c. Some articles mentions that batch normalizing can partially solve that problem as well.
"""


part3_q3 = r"""
This experiment also shows that lower depth give better results like we have seen on experiment 1 for **L=2**.
In addition to that, this experiment also shows that deeper depths are not trainable too. (**L=8**)
So we can see that our conclusions are consistent with the former experiment.
A new conclusion from this experiment is that within a constant depth - we can see better results for a higher number of filters **K**.
It happens because higher number of filters allows better features extractions from an image.
"""

part3_q4 = r"""
As we saw on former experiments increasing depth causes the network to perform worse.
As we keep using the convClassifier which has no answer to the vanishing gradient problem we would not be able to succeed on deeper networks.
"""


part3_q5 = r"""
Using the ResNet we can witness that it solves our former problem and we are able to train deeper networks.
We can witness an improvement in accuracy and less overfitting opposed to the convClassifier.
So in comparison to the former experiments we can see that deeper **L** values are now trainable and we get better results for increasing number of channels.
"""

part3_q6 = r"""
1.  We have used 7X7 conv for first layer and then used inception blocks (got the idea from d2l.ai book online) and used batch norm + activation + pooling every few layers.

2. In comparison to the experiments using the ConvClassifier the YCN is doing much better, but slightly worse than the Resnet in our opinion.
also we can see that this net is doing better on deeper depths than the convClassifier.
"""
# ==============
