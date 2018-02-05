# Oil-spill-challenge-MLP
MLP code

### We have a fully connected MLP with ReLU activation for the hidden nodes and softmax at the output.
1. The input is a 784x1 vector and hence we choose the hidden layer to be input*output size which is nearly 10000 nodes.
2. The batch size is 30, since it gives us 2000 epochs to check out output on.
3. Learning rate is a standard at 0.01, with decay rate.

We get a 96.5% training accuracy, which is nearly the same as in Yann Le Cunn's paper.
