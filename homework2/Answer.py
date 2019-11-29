import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


torch.manual_seed(1021)
torch.cuda.manual_seed(1021)
np.random.seed(1021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print('cuda is available')

#########################################################################################################
# ----------------------------------------- SELECT DATA & TASK -----------------------------------------#
# data = {cifar, mnist}
# task = {1_imbalanced, 2_semisupervised, 3_noisy}

data = 'mnist'
task = '1_imbalanced'
# task = '2_semisupervised'
# task = '3_noisy'
# ----------------------------------------- END OF SELECTION -------------------------------------------#
#########################################################################################################

def load_dataset():
    # check the data & task you selected
    print('#'*50)
    print('DATA: {}'.format(data))
    print('TASK: {}'.format(task))
    print('#'*50)

    # Load the dataset
    data_path = os.path.join('./data', data, task)
    x_train = np.load(os.path.join(data_path, 'train_x.npy'))
    y_train = np.load(os.path.join(data_path, 'train_y.npy'))
    x_valid = np.load(os.path.join(data_path, 'valid_x.npy'))
    y_valid = np.load(os.path.join(data_path, 'valid_y.npy'))
    x_test = np.load(os.path.join(data_path, 'test_x.npy'))

    # Check the shape of your data
    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('x_valid shape: {}'.format(x_valid.shape))
    print('y_valid shape: {}'.format(y_valid.shape))
    print('x_test shape: {}'.format(x_test.shape))
    print('#'*50)

    return x_train, y_train, x_valid, y_valid, x_test


#########################################################################################################
# ----------------------------------------- SAMPLE CODE ------------------------------------#
# No need to use the code below.

class SampleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SampleModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

class MnistResNet(nn.Module):
    def __init__(self):
        super(MnistResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        self.resnet18 = models.resnet18()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        shape = x.shape
        x = x.reshape([shape[0], shape[1], 28, 28])
        x = self.conv1(x)
        x = self.resnet18(x)
        return x

def train(x_train, y_train, x_valid, y_valid, x_test, batch_size=100, lr=0.1, num_epochs=100, print_every=10):
    # hidden_size = 100
    train_acc = []
    valid_acc = []

    num_feature = x_train.shape[1]
    model = MnistResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2) 

    num_data = x_train.shape[0]
    num_batch = int(np.ceil(num_data / batch_size))

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_valid = torch.from_numpy(x_valid)
    y_valid = torch.from_numpy(y_valid)
    x_test = torch.from_numpy(x_test)

    # sample example (w/o stopping condition). You can adjust stopping condition for your own
    for i in range(num_epochs):
        epoch_loss = 0.0
        
        for b in range(0, len(x_train), batch_size):
            x_batch = x_train[b: b + batch_size]
            y_batch = y_train[b: b + batch_size]

            images = Variable(x_batch).to(device)
            labels = Variable(y_batch).to(device)

            outputs = model(images)
            loss = criterion(outputs, torch.max(labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= num_batch
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(i + 1, num_epochs, b + 1, num_batch, epoch_loss))

        # Train accuracy
        pred = model(x_train)
        pred = np.argmax(pred.data.cpu().numpy(), axis=1)
        true = np.argmax(y_train.data.cpu().numpy(), axis=1)

        total = x_train.shape[0]
        correct = len(np.where(pred == true)[0])
        tr_acc = correct / total
        train_acc.append(tr_acc)

        # Valid accuracy
        pred = model(x_valid)
        pred = np.argmax(pred.data.cpu().numpy(), axis=1)
        true = np.argmax(y_valid.data.cpu().numpy(), axis=1)

        total = x_valid.shape[0]
        correct = len(np.where(pred == true)[0])
        va_acc = correct / total
        valid_acc.append(va_acc)

        if i % print_every == 0:
            print('[EPOCH %d] Loss = %f' % (i, epoch_loss))
            print('Train Accuracy = %.4f' % tr_acc)
            print('Valid Accuracy = %.4f' % va_acc)
        train_scheduler.step(i)

    # Predict the test (sample example)
    test_pred = model(x_test)
    test_pred = np.argmax(test_pred.data.cpu().numpy(), axis=1)

    # ----------------------------------------- END OF CODE-------------------------------------------#
    #########################################################################################################

    # You need to save and submit the 'test_pred.npy' under the output directory.
    save_path = os.path.join('./output', data, task)
    np.save(os.path.join(save_path, 'test_pred.npy'), test_pred)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    args = parser.parse_args()

    x_train, y_train, x_valid, y_valid, x_test = load_dataset()

    train(x_train, y_train, x_valid, y_valid, x_test, lr=args.lr)

    