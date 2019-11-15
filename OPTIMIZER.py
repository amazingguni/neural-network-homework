import time
import numpy as np
import matplotlib.pyplot as plt
from Answer import OneLayerNet
from utils import load_titanic

loss = {'train_sgd': [], 'test_sgd': [], 'train_momentum': [], 'test_momentum': [], 'train_adam':[], 'test_adam':[]}
#########################################################################################################
# ------------------------------------------WRITE YOUR CODE----------------------------------------------#
hidden = None
num_epochs = None

learning_rate = None
beta1 = None
beta2 = None
mu = None
# -----------------------------------------END OF YOUR CODE----------------------------------------------#
#########################################################################################################
epsilon = 1e-8
print_every = 100
batch_size = 64
x_train, y_train, x_test, y_test = load_titanic('./data')

num_feature = x_train.shape[1]
num_traindata = len(x_train)
num_testdata = len(x_test)
num_batch = int(np.ceil(num_traindata / batch_size))

def train_net(model):
    train_loss, test_loss = [], []

    for i in range(1, num_epochs + 1):
        for b in range(0, len(x_train), batch_size):
            x_batch = x_train[b: b + batch_size]
            y_batch = y_train[b: b + batch_size]

            # train
            model.train(x_batch, y_batch, mu, beta1, beta2, epsilon)

            # predict
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)

            # calculate the loss
            each_train_loss = model.loss(y_pred_train, y_train)
            each_test_loss = model.loss(y_pred_test, y_test)

        if i % print_every == 0:
            print('Epoch {} | Train: {:.3f} Test Loss: {:.3f}'.format(i, each_train_loss, each_test_loss))

        train_loss.append(each_train_loss)
        test_loss.append(each_test_loss)
    return train_loss, test_loss

def accuracy(model):
    pred = model.predict(x_test)
    correct = 0
    for i in range(num_testdata):
        if pred[i] < 0.5 and y_test[i] == 0:
            correct += 1
        elif pred[i] >= 0.5 and y_test[i] == 1:
            correct +=1
    return correct / num_testdata


model_sgd = OneLayerNet(input_size=num_feature, hidden_size=hidden,
                        output_size=1, optimizer='sgd', learning_rate=learning_rate)
model_momentum = OneLayerNet(input_size=num_feature, hidden_size=hidden,
                             output_size=1, optimizer='momentum', learning_rate=learning_rate)
model_adam = OneLayerNet(input_size=num_feature, hidden_size=hidden,
                         output_size=1, optimizer='adam', learning_rate=learning_rate)

loss['train_sgd'], loss['test_sgd'] = train_net(model_sgd)
loss['train_momentum'], loss['test_momentum'] = train_net(model_momentum)
loss['train_adam'], loss['test_adam'] = train_net(model_adam)

plt.plot(loss['train_sgd'], label='SGD Train Loss')
plt.plot(loss['test_sgd'], label='SGD Test Loss')
plt.plot(loss['train_momentum'], label='Momentum Train Loss')
plt.plot(loss['test_momentum'], label='Momentum Test Loss')
plt.plot(loss['train_adam'], label='Adam Train Loss')
plt.plot(loss['test_adam'], label='Adam Test Loss')
plt.title('Train & Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('SGD Accuracy: {:.2f}'.format(accuracy(model_sgd)))
print('Momentum Accuracy: {:.2f}'.format(accuracy(model_momentum)))
print('Adam Accuracy: {:.2f}'.format(accuracy(model_adam)))