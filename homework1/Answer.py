import numpy as np
np.random.seed(1000)

def sign(x):
    """
    #####################################################################################################
    # TODO: Implement the sign function for Perceptron.
    #
    #       - input (x) : input for sign function
    #       - return (sx) : sign value for x = 1 if x > 0, -1 otherwise
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """

    sx = None
    #########################################################################################################
    # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
    sx = np.vectorize(lambda item: 1 if item > 0 else -1)(x)
    # -----------------------------------------END OF YOUR CODE----------------------------------------------#
    #########################################################################################################

    return sx


class Perceptron:
    def __init__(self, num_features):
        self.W = np.random.rand(num_features, 1)
        self.b = np.random.rand(1)

    def forward(self, x):
        """
        #####################################################################################################
        # TODO: Implement the forward process of a single neuron (perceptron).
        # input(x) ---> (weight, bias) ---> z ---> (sign function) ---> sign(z)
        #
        #       - input (x) : input for perceptron; numpy array of (N, D)
        #       - return (out) : output of perceptron; numpy array of (N, 1)
        #
        # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
        #####################################################################################################
        """
        out = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        z = np.dot(x, self.W) + self.b
        out = sign(z)
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        return out

    def train(self, x, y, lr):
        num_data = x.shape[0]
        """
        #####################################################################################################
        # TODO: Implement the batch training of perceptron.
        # Perceptron batch training updates weights all at once for every data
        # Please be careful not to implement other functions such as shuffling data, etc.
        #       - inputs:
        #           x : input for perceptron; numpy array of (N, D)
        #           y : output (label) of input x; numpy array of (N, )
        #           lr : learning rate
        #       - return : None
        #
        # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
        #####################################################################################################
        """

        while True:

            dW = np.zeros_like(self.W) # gradients of Weight (W)
            db = np.zeros_like(self.b) # gradients of bias (b)

            # Repeat until quit condition is satisfied.
            quit = True
            for i in range(num_data):
            #########################################################################################################
            # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
                y_hat = self.forward(x).squeeze()
                if np.all(np.equal(y, y_hat)):
                    break
                quit = False
                for x_each, y_each, y_hat_each in zip(x, y, y_hat):
                    if y_each == y_hat_each:
                        continue
                    dW += np.expand_dims(x_each * y_each * lr, axis=1)
                    db += np.expand_dims(1 * y_each * lr, axis=1)
            self.W += dW
            self.b += db    
            # -----------------------------------------END OF YOUR CODE----------------------------------------------#
            #########################################################################################################
            if quit:
                break


class Sigmoid:
    """
    #####################################################################################################
    # TODO: Implement the forward and backward process of Sigmoid Function.
    # Please consider that the results from the forward function will be used in the backward function.
    #
    # (1) forward():
    #       - input (x) : sigmoid input in any shape
    #       - return (self.out) : sigmoid output = sigmoid(x)
    #
    # (2) backward():
    #       - input (dout) : delta from backpropagation until now
    #       - return (dx) : gradients w.r.t sigmoid input x
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.out = 1/(1+np.exp(-x))
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return self.out

    def backward(self, dout):
        dx = None

        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        dx = dout * (1.0 - self.out) * self.out
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return dx



class ReLU:
    """
    #####################################################################################################
    # TODO: Implement the forward and backward process of ReLU Function.
    # Please consider that the mask information from the forward function will be used in the backward function.
    #
    # (1) forward():
    #       - input (x) : ReLU input in any shape
    #       - return (out) : ReLU output = ReLU(x)
    #
    # (2) backward():
    #       - input (dout) : delta from backpropagation until now
    #       - return (dx) : gradients w.r.t ReLU input x
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """

    def __init__(self):
        self.relu_mask = None

    def forward(self, x):
        out = None

        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.relu_mask = (x <= 0)
        out = np.maximum(0, x)
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return out

    def backward(self, dout):
        dx = None

        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        dout[self.relu_mask] = 0
        dx = dout
        
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return dx

def softmax(x):
    """
    #####################################################################################################
    # TODO: Implement the Softmax Function.
    # Please be careful not to make NaN (prevent overflow).
    #
    #       - input (x) : a vector with dimension (N, D)
    #       - return (softmax_output) : softmax output = softmax(x)
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """
    # #########################################################################################################
    # # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
    max_x = np.max(x, axis=1)
    max_x = np.expand_dims(max_x, axis=1)
    exps = np.exp(x - max_x)
    
    s = np.sum(exps, axis=1)
    s = np.expand_dims(s, axis=1)
    softmax_output = exps / s
    # # -----------------------------------------END OF YOUR CODE----------------------------------------------#
    # #########################################################################################################

    return softmax_output


class InputLayer:
    """
    #####################################################################################################
    # TODO: Implement the forward and backward process of Input Layer.
    #
    # (1) forward():
    #       - input (x) : input vector with dimension (N, D)
    #       - return (self.out) : output vector with dimension (N, H)
    #
    # (2) backward():
    #       - input (dout) : delta from backpropagation until now
    #       - return : None
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """
    def __init__(self, num_features, hidden1, activation):
        
        # Weights and bias
        self.W = np.random.rand(num_features, hidden1)
        self.b = np.zeros(hidden1)

        # Gradient of Weights and bias
        self.dW = None
        self.db = None

        # Forward input
        self.x = None

        # Activation function (Sigmoid or ReLU)
        self.act = activation()

    def forward(self, x):

        self.x = None
        self.out = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.x = x
        z = np.dot(self.x,self.W) + self.b
        self.out = self.act.forward(z)
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return self.out

    def backward(self, dout):

        self.dW = None
        self.db = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        
        dx = np.dot(self.act.backward(dout), self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################


class HiddenLayer:
    """
    #####################################################################################################
    # TODO: Implement the forward and backward process of Hidden Layer.
    # Please save self.dW, self.db to update later (in the backward process)!
    #
    # (1) forward():
    #       - input (x) : input vector with dimension (N, H1)
    #       - return (self.out) : output vector with dimension (N, H2)
    #
    # (2) backward():
    #       - input (dprev) : delta from backpropagation until now
    #       - return (dx) : gradients of hidden layer input x (NOT MLP Input x)
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """
    def __init__(self, hidden1, hidden2):

        # Weights and bias
        self.W = np.random.rand(hidden1, hidden2)
        self.b = np.zeros(hidden2)

        # Gradient of Weights and bias
        self.dW = None
        self.db = None

        # ReLU function
        self.act = ReLU()

    def forward(self, x):

        self.x = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.x = x
        z = np.dot(self.x, self.W) + self.b
        self.out = self.act.forward(z)
        # print(self.out[0])
        
        # print(f'Hidden x {x.shape}')

        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return self.out

    def backward(self, dout):

        dx = None
        self.dW = None
        self.db = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        dx = np.dot(self.act.backward(dout), self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return dx


class OutputLayer:
    """
    #####################################################################################################
    # TODO: Implement the (1) cross_entropy loss function and (2) predict & (3) backward process of Output Layer.
    #
    # (1) cross_entropy_loss(): Please add 'epsilon' to prevent 0 not to be an input to the log function.
    #       - inputs
    #           y_pred : predicted label
    #           y : real label
    #       - return (ce_loss) : cross entropy loss value
    #
    # (2) predict():
    #       - input (x) : input data
    #       - return (y_pred) : predicted label
    #
    # (3) backward(): Please save self.dW and self.db to update later!
    #       - input (dout) : delta from backpropagation until now
    #       - return (dx) : gradients of output layer input x (NOT MLP Input x)
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """
    def __init__(self, hidden2, num_outputs):

        # Weights and bias
        self.W = np.random.rand(hidden2, num_outputs)
        self.b = np.zeros(num_outputs)

        # Gradient of Weights and bias
        self.dW = None
        self.db = None

        # Input (x), label(y), prediction(y_pred)
        self.x = None
        self.y = None
        self.y_pred = None

        # Loss
        self.loss = None

    def forward(self, x, y):
        self.y_pred = self.predict(x) # prediction of labels
        self.y = y
        self.x = x
        self.loss = self.cross_entropy_loss(self.y_pred, self.y) # calculation of cross-entropy loss
        return self.loss

    def cross_entropy_loss(self, y_pred, y):

        epsilon = 1e-10
        ce_loss = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        batch_size = self.y.shape[0]
        ce_loss = -np.sum(np.log(y_pred + epsilon) * y) / batch_size
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return ce_loss

    def predict(self, x):

        y_pred = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        y_pred = softmax(np.dot(x, self.W) + self.b)
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return y_pred

    def backward(self, dout=1):

        batch_size = self.y.shape[0]
        dx = None
        #########################################################################################################
        # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
        delta = self.y_pred
        delta[range(batch_size), self.y.argmax(axis=1)] -= 1
        delta /= batch_size

        dx = np.dot(delta, self.W.T) 
        self.dW = np.dot(self.x.T, delta)
        self.db = np.sum(delta, axis = 0)
        # -----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        
        return dx




class OneLayerNet:
    """
    #####################################################################################################
    # TODO: Implement two optimizers : (1) SGD and (2) ADAM
    # Please consider that there are two weights (self.W1 and self.W2) to update and use self.dw1 and self.dw2 (=gradients).
    # Please note that beta1 = decay_rate1 and beta2 = decay_rate2 and there is no bias term.
    #
    # (1) SGD: Please use the following hyperparameter: learning_rate
    #
    # (2) MOMENTUM: Please use the following hyperparameters: learning_rate, mu
    #
    # (3) ADAM: Please use the following hyperparameters: learning rate, beta1, beta2, epsilon
    #           Please implement the ADAM as explained in the paper. (Reference: https://arxiv.org/pdf/1412.6980.pdf)
    #
    #    - Update biased first moment
    #    - Update biased second moment
    #    - Compute bias-corrected first moment
    #    - Compute bias-corrected second moment
    #    - Update parameters
    #
    # NOTE: Write your answers only under the 'WRITE YOUR CODE'.
    #####################################################################################################
    """
    def __init__(self, input_size, hidden_size, output_size, optimizer, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.W1 = np.random.rand(input_size, hidden_size)   # learning parameters to update
        self.W2 = np.random.rand(hidden_size, output_size)  # learning parameters to update

        self.dW1 = None             # gradient of W1
        self.dw2 = None             # gradient of W2

        self.velocity1 = 0              # velocity for W1 (will be used in Momentum optimizer)
        self.velocity2 = 0              # velocity for W2 (will be used in Momentum optimizer)

        self.time_step = 0          # time step (will be used in ADAM optimizer)
        self.first_momentum1 = 0    # first momentum for W1 (will be used in ADAM optimizer)
        self.first_momentum2 = 0    # first momentum for W2 (will be used in ADAM optimizer)
        self.second_momentum1 = 0   # second momentum for W1 (will be used in ADAM optimizer)
        self.second_momentum2 = 0   # second momentum for W2 (will be used in ADAM optimizer)

    def forward(self, x, w):
        return  1 / (1 + np.exp(-np.dot(x, w)))

    def predict(self, x):
        z1 = self.forward(x, self.W1)
        z2 = self.forward(z1, self.W2)
        return z2

    def loss(self, y_pred, y_real):
        return np.mean((y_pred - y_real)**2)

    def train(self, x, y, mu=None, beta1=None, beta2=None, epsilon=None):
        # Forward
        z = self.forward(x, self.W1)
        y_pred = self.forward(z, self.W2)

        # Backward
        delta2 = (y_pred-y) * y_pred * (1-y_pred)
        delta1 = np.dot(delta2, self.W2.T) * z * (1-z)

        # Gradients of W2 and W1; will be used in SGD and ADAM optimizer
        self.dW2 = np.dot(z.T, delta2)
        self.dW1 = np.dot(x.T, delta1)
        if self.optimizer =='sgd':
            #########################################################################################################
            # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
            self.W2 = self.W2 - self.learning_rate * self.dW2
            self.W1 = self.W1 - self.learning_rate * self.dW1
            # -----------------------------------------END OF YOUR CODE----------------------------------------------#
            #########################################################################################################


        elif self.optimizer == 'momentum':
            #########################################################################################################
            # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
            self.velocity2 = mu * self.velocity2 - self.learning_rate * self.dW2
            self.velocity1 = mu * self.velocity1 - self.learning_rate * self.dW1

            self.W2 = self.W2 + self.velocity2
            self.W1 = self.W1 + self.velocity1
            # self.W2 = self.W2 - self.learning_rate * self.dW2
            # self.W1 = self.W1 - self.learning_rate * self.dW1
            # -----------------------------------------END OF YOUR CODE----------------------------------------------#
            #########################################################################################################


        elif self.optimizer == 'adam':
            self.time_step += 1
            #########################################################################################################
            # ------------------------------------------WRITE YOUR CODE----------------------------------------------#
            self.first_momentum1 = beta1 * self.first_momentum1 + (1 - beta1) * self.dW1
            self.second_momentum1 = beta2 * self.second_momentum1 + (1 - beta2) * np.power(self.dW1, 2)
            corrected_first_momentum1 = self.first_momentum1 / (1 - (beta1 ** self.time_step))
            corrected_second_momentum1 = self.second_momentum1 / (1 - (beta2 ** self.time_step))
            self.W1 = self.W1 - self.learning_rate * (1 / np.sqrt(corrected_second_momentum1 + epsilon)) * corrected_first_momentum1

            self.first_momentum2 = beta1 * self.first_momentum2 + (1 - beta1) * self.dW2
            self.second_momentum2 = beta2 * self.second_momentum2 + (1 - beta2) * np.power(self.dW2, 2)
            corrected_first_momentum2 = self.first_momentum2 / (1 - (beta1 ** self.time_step))
            corrected_second_momentum2 = self.second_momentum2 / (1 - (beta2 ** self.time_step))
            self.W2 = self.W2 - self.learning_rate * (1 / np.sqrt(corrected_second_momentum2 + epsilon)) * corrected_first_momentum2
            # -----------------------------------------END OF YOUR CODE----------------------------------------------#
            #########################################################################################################
        else:
            raise NotImplementedError