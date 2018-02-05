
# coding: utf-8

# In[3]:


# Author : Srinath
# For cerebras challenge
# Credits : FlorianMuellerklein


# In[12]:


from __future__ import division
import random
import numpy as np
from sklearn.preprocessing import scale

np.seterr(all = 'ignore')


# In[13]:


def sigmoid(x):
    '''
    Takes a real input and computes its sigmoid
    :param x : Real input
    :type x : float
    '''
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    '''
    Returns the derivative of a sigmoid, given the sigmoid value
    :param y: Sigmoid of a real number
    :type y: float
    '''
    return y * (1.0 - y)


# In[14]:


def relu(x):
    '''
    Returns ReLU of x
    '''
    return x * (x > 0)

def drelu(x):
    '''
    Returns derivative of ReLU of x
    '''
    return 1. * (x > 0)


# In[15]:


def softmax(w):
    '''
    Returns the softmax output
    :param w: Input
    '''
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist


# In[16]:


class MLP_Classifier(object):
    """
    Basic MultiLayer Perceptron (MLP) neural network with regularization and learning rate decay
    Consists of three layers: input, hidden and output.
    """
    
    def __init__(self, input, hidden, output, iterations = 50, learning_rate = 0.01, 
                l2_in = 0, l2_out = 0, momentum = 0, rate_decay = 0, 
                output_layer = 'softmax', verbose = True):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        :param iterations: how many epochs
        :param learning_rate: initial learning rate
        :param output_layer: activation (transfer) function of the output layer
        :param verbose: whether to spit out error rates while training
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_in = l2_in
        self.l2_out = l2_out
        self.momentum = momentum
        self.rate_decay = rate_decay
        self.verbose = verbose
        self.output_activation = output_layer
        
        # initialize arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden 
        self.output = output

        # set up array of 1s for activations
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)

        # create randomized weights
        input_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs!')

        # input activations
        self.ai[0:self.input -1] = inputs

        # hidden activations
        sum = np.dot(self.wi.T, self.ai)
        self.ah = relu(sum)
        
        # output activations
        sum = np.dot(self.wo.T, self.ah)
        self.ao = softmax(sum)
        
        return self.ao

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        
        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        output_deltas = -(targets - self.ao)
        
        # calculate error terms for hidden
        # delta (theta) tells you which direction to change the weights
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = drelu(self.ah) * error
        
        # update the weights connecting hidden to output, change == partial derivative
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0],1))
        regularization = self.l2_out * self.wo
        self.wo -= self.learning_rate * (change + regularization) + self.co * self.momentum 
        self.co = change 

        # update the weights connecting input to hidden, change == partial derivative
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        regularization = self.l2_in * self.wi
        self.wi -= self.learning_rate * (change + regularization) + self.ci * self.momentum 
        self.ci = change

        # calculate error
        error = -sum(targets * np.log(self.ao))
        
        return error

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))

    def fit(self, patterns):
        if self.verbose == True:
            print 'Using softmax activation in output layer'
      
        num_example = np.shape(patterns)[0]
                
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            
            if i % 10 == 0 and self.verbose == True:
                error = error/num_example
                print('Training error %-.5f' % error)
                
            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
                
    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions


# In[17]:


def load_data():
    data = np.loadtxt('Data/mnist_train.csv', delimiter = ',')

    # first ten values are the one hot encoded y (target) values

    temp = np.array(data[:,0],dtype=np.int)
    y = np.zeros((data.shape[0],10))
    y[np.arange(data.shape[0]), temp]=1

    data = data[:,1:] # x data
    data = scale(data)
    out = []
    
    # populate the tuple list with the data
    for i in range(data.shape[0]):
        tupledata = list((data[i,:].tolist(), y[i].tolist()))
        out.append(tupledata)

    return out


# In[ ]:


X = load_data()
NN = MLP_Classifier(784, 40000, 10, iterations = 50, learning_rate = 0.01, 
                    momentum = 0.5, rate_decay = 0.0001, 
                    output_layer = 'softmax')
NN.fit(X)

