import numpy as np
from utils import load_iris
from Answer import Perceptron

x_train, y_train, x_test, y_test = load_iris('./data')

num_features = x_train.shape[1]
perceptron = Perceptron(num_features)
learning_rate = None

#########################################################################################################
# ------------------------------------------WRITE YOUR CODE----------------------------------------------#
learning_rate = 0.1
# -----------------------------------------END OF YOUR CODE----------------------------------------------#
#########################################################################################################

perceptron.train(x_train, y_train, learning_rate)
y_pred = perceptron.forward(x_test)

num_correct = len(np.where(y_pred.reshape(-1) == y_test)[0])
num_total = len(y_test)

print('Test Accuracy : %.2f' % (num_correct / num_total))
