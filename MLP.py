import numpy as np
import matplotlib.pyplot as plt
from Answer import ReLU, InputLayer, HiddenLayer, OutputLayer
from utils import load_fashion_mnist

class MLP:
    def __init__(self, input_size, hidden1, hidden2, output_size):
        self.input_layer = InputLayer(input_size, hidden1, ReLU)
        self.hidden_layer = HiddenLayer(hidden1, hidden2)
        self.output_layer = OutputLayer(hidden2, output_size)

    def predict(self, x):
        x = self.input_layer.forward(x)
        x = self.hidden_layer.forward(x)
        prob = self.output_layer.predict(x)
        pred = np.argmax(prob, axis=-1)
        return pred

    def loss(self, x, y):
        x = self.input_layer.forward(x)
        x = self.hidden_layer.forward(x)
        loss = self.output_layer.forward(x, y)
        return loss

    def gradient(self):
        dout = 1
        dout = self.output_layer.backward(dout=dout)
        dout = self.hidden_layer.backward(dout)
        self.input_layer.backward(dout)

    def update(self, learning_rate):
        self.input_layer.W -= self.input_layer.dW * learning_rate
        self.input_layer.b -= self.input_layer.db * learning_rate
        self.hidden_layer.W -= self.hidden_layer.dW * learning_rate
        self.hidden_layer.b -= self.hidden_layer.db * learning_rate
        self.output_layer.W -= self.output_layer.dW * learning_rate
        self.output_layer.b -= self.output_layer.db * learning_rate


#########################################################################################################
# ------------------------------------------WRITE YOUR CODE----------------------------------------------#
hidden1 = None
hidden2 = None
num_epochs = None
learning_rate = None
# -----------------------------------------END OF YOUR CODE----------------------------------------------#
#########################################################################################################
print_every = 10
batch_size = 100
train_acc = []
test_acc = []

x_train, y_train, x_test, y_test = load_fashion_mnist('./data')

num_feature = x_train.shape[1]
model = MLP(input_size=num_feature, hidden1=hidden1, hidden2=hidden2, output_size=10)

num_data = len(x_train)
num_batch = int(np.ceil(num_data / batch_size))

for i in range(1, num_epochs + 1):
    epoch_loss = 0.0
    for b in range(0, len(x_train), batch_size):
        x_batch = x_train[b: b + batch_size]
        y_batch = y_train[b: b + batch_size]

        loss = model.loss(x_batch, y_batch)
        epoch_loss += loss

        model.gradient()
        model.update(learning_rate)
    epoch_loss /= num_batch

    # Train accuracy
    pred = model.predict(x_train)
    true = np.argmax(y_train, axis=1)

    total = len(x_train)
    correct = len(np.where(pred == true)[0])
    tr_acc = correct / total
    train_acc.append(tr_acc)

    # Test accuracy
    pred = model.predict(x_test)
    true = np.argmax(y_test, axis=1)

    total = len(x_test)
    correct = len(np.where(pred == true)[0])
    te_acc = correct / total
    test_acc.append(te_acc)

    if i % print_every == 0:
        print('[EPOCH %d] Loss = %f' % (i, epoch_loss))
        print('Train Accuracy = %.3f' % tr_acc)
        print('Test Accuracy = %.3f' % te_acc)

x_axis = list(range(len(test_acc)))
plt.plot(x_axis, train_acc, 'b-', label='Train Acc.')
plt.plot(x_axis, test_acc, 'r-', label='Test Acc.')
plt.title('Train & Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()