import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, CIFAR100
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_

from models.resnet import resnet18
torch.manual_seed(1021)
torch.cuda.manual_seed(1021)
np.random.seed(1021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f'cuda is available: {torch.cuda.current_device()}')

#########################################################################################################
# ----------------------------------------- SELECT DATA & TASK -----------------------------------------#
# data = {cifar, mnist}
# task = {1_imbalanced, 2_semisupervised, 3_noisy}

#data = 'mnist'
# data = 'cifar'
#task = '1_imbalanced'
# task = '2_semisupervised'
# task = '3_noisy'
# ----------------------------------------- END OF SELECTION -------------------------------------------#
#########################################################################################################


class HomeworkDataset(Dataset):
    def __init__(self, x_data, y_data, data, transform=None):
        self.x_data = x_data
        self.y_data = y_data 
        self.len = self.x_data.shape[0]
        self.data = data
        self.transform = transform
        self.num_of_classes = 100 if data == 'cifar' else 10
        
    def __getitem__(self, index):
        image = self.x_data[index]
        if self.data == 'cifar':
            image = np.uint8(image.reshape(3, 32, 32) * 255).transpose(1, 2, 0)
            image = Image.fromarray(image, 'RGB')
        elif self.data == 'mnist':
            image = image.reshape(28, 28)
            image = Image.fromarray(np.uint8(image * 255) , 'L')
            #image = image.convert(mode='RGB')
        if self.transform:
            image = self.transform(image)
        label = self.y_data[index]
        return image, label

    def __len__(self):
        return self.len

class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network1 = resnet18()
        self.network2 = resnet18()
        self.network3 = resnet18()
        self._reset_parameters()

    def forward(self, image):
        output1 = self.network1(image)
        output2 = self.network2(image)
        output3 = self.network3(image)
        return output1 + output2 + output3

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

def draw(train_acc, test_acc, save_path):
    x_axis = list(range(len(test_acc)))
    plt.plot(x_axis, train_acc, 'b-', label='Train Acc.')
    plt.plot(x_axis, test_acc, 'r-', label='Test Acc.')
    plt.title('Train & Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/plot.png')
    plt.clf()

def train(model, train_loader, valid_loader, batch_size, lr, num_epochs, print_every=5, checkpoint_path=None, output_path=None):
    train_acc = []
    valid_acc = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2) 
    train_begin = time.time()
    best_acc = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_begin = time.time()
        model.train()
        for batch_data in train_loader:
            images, labels = batch_data 
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += torch.sum(pred.eq(labels)).item()
            total += outputs.shape[0]
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= total

        # Train accuracy
        tr_acc = correct / total
        train_acc.append(tr_acc)

        correct = 0
        total = 0
        # Valid accuracy
        model.eval()
        for images, labels in valid_loader:
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += torch.sum(pred.eq(labels)).item()
            total += outputs.shape[0]
        va_acc = correct / total
        valid_acc.append(va_acc)
        if best_acc < va_acc and checkpoint_path:
            torch.save(model.state_dict(), f'{checkpoint_path}/resnet18-{epoch}-{va_acc*100:.2f}.pth')
            best_acc = va_acc
        if epoch % print_every == 0:
            current = time.time()
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 60.0
            print(f'[EPOCH {epoch}] Loss = {epoch_loss:.4f} elapsed: epoch({epoch_elapsed:.3f}m) total({train_elapsed:.3f}m)')
            print(f'Train Accuracy = {tr_acc * 100:.4f}')
            print(f'Valid Accuracy = {va_acc * 100:.4f}({correct}/{total})')
        train_scheduler.step(epoch)
    draw(train_acc, valid_acc, output_path)

def test(model, test_loader, output_path, data):
    model.eval()
    total_pred = np.array([])
    for images, _ in test_loader:
        images = Variable(images).to(device)
        test_pred = model(images)
        test_pred = np.argmax(test_pred.data.cpu().numpy(), axis=1)
        total_pred = np.hstack([total_pred, test_pred])

    true = np.load(f'{data}_test_y.npy')
    total = true.shape[0]
    correct = len(np.where(total_pred == true)[0])
    real_acc = correct / total
    #print(','.join([f'({pred}, {label})' for pred, label in zip(total_pred, true)]))
    print(f'Real Accuracy = {real_acc * 100:.4f}({correct}/{total})')

    # You need to save and submit the 'test_pred.npy' under the output directory.
    np.save(os.path.join(output_path, 'test_pred.npy'), test_pred)

def make_balanced_classes(x_data, y_data, data):
    num_of_classes = 10 if data == 'mnist' else 100
    count = [0] * num_of_classes
    for label in y_data:
        count[label] += 1
    max_count = max(count)
    min_count = min(count)
    if max_count == min_count:
        return x_data, y_data
    ratio = int(max_count/min_count)
    unbalanced_x = []
    unbalanced_y = []

    for x, y in zip(x_data, y_data):
        if count[y] != min_count:
            continue
        unbalanced_x += [x] * (ratio - 1)
        unbalanced_y += [y] * (ratio - 1)
    x_data = np.vstack((x_data, unbalanced_x))
    y_data = np.hstack((y_data, unbalanced_y))
    return x_data, y_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=200, help='num_epochs')
    parser.add_argument('--data', default='mnist', help='mnist, cifar')
    parser.add_argument('--task', default='1_imbalanced', help='1_imbalanced, 2_semisupervised, 3_noisy, origin')
    parser.add_argument('--print', type=int, default=10, help='Print every..')
    parser.add_argument('--use-all', action='store_true', help='True if you want to use both valid and train for training')
    parser.add_argument('--balanced', action='store_true', help='True if you want to balance data set')


    args = parser.parse_args()

    data = args.data
    task = args.task
    print('#'*50)
    print('DATA: {}'.format(data))
    print('TASK: {}'.format(task))
    print(f'use all: {args.use_all}')
    print(f'epoch: {args.epoch}')
    print(f'batch_size: {args.batch_size}')
    print('#'*50)
    
    means = (0.5, 0.5, 0.5)
    deviations = means

    crop_size = 32 if data == 'cifar' else 28

    common_compose = [
        transforms.ToTensor(),
        transforms.Normalize(means, deviations) 
    ]

    if data == 'mnist':
        common_compose = [transforms.Grayscale(3)] + common_compose
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ] + common_compose)
    
    transform_test = transforms.Compose(common_compose)
    
    
    batch_size = args.batch_size

    # Load the dataset
    if task == 'origin':
        origin_dataset = CIFAR100 if data == 'cifar' else FashionMNIST
        train_dataset = origin_dataset(root='./data_orig', train=True, download=True, transform=transform_train)
        valid_dataset = origin_dataset(root='./data_orig', train=False, download=True, transform=transform_test)
        test_dataset = origin_dataset(root='./data_orig', train=False, download=True, transform=transform_test)
    else:    
        data_path = os.path.join('./data', data, task)
        train_x_data = np.load(os.path.join(data_path, 'train_x.npy'))
        valid_x_data = np.load(os.path.join(data_path, 'valid_x.npy'))
        test_x_data = np.load(os.path.join(data_path, 'test_x.npy'))

        train_y_data = np.argmax(np.load(os.path.join(data_path, 'train_y.npy')), axis=1)
        valid_y_data = np.argmax(np.load(os.path.join(data_path, 'valid_y.npy')), axis=1)
        test_y_data = np.load(f'{data}_test_y.npy')
        if args.balanced:
            train_x_data, train_y_data = make_balanced_classes(train_x_data, train_y_data, data)

        if args.use_all:
            train_x_data = np.concatenate((train_x_data, valid_x_data), axis=0)
            train_y_data = np.concatenate((train_y_data, valid_y_data), axis=0)
            valid_x_data = test_x_data
            valid_y_data = test_y_data
        train_dataset = HomeworkDataset(train_x_data, train_y_data, data, transform_train)
        valid_dataset = HomeworkDataset(valid_x_data, valid_y_data, data, transform_test)
        test_dataset = HomeworkDataset(test_x_data, test_y_data, data, transform_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    

    # Check the shape of your data
    print(f'x_train shape: {len(train_dataset)} {train_dataset[0][0].shape}')
    print(f'x_valid shape: {len(valid_dataset)} {valid_dataset[0][0].shape}')
    print(f'x_test shape: {len(test_dataset)} {test_dataset[0][0].shape}')
    print('#'*50)

    #model = models.resnet18(pretrained=True).to(device)
    #model = models.resnet18().to(device)
    #model = models.shufflenet_v2_x1_0().to(device)

    model = resnet18().to(device)
    # model = EnsembleModel().to(device)
    if device == 'cuda':
        print('Turn on DataParallel')
        model = torch.nn.DataParallel(model)
        
    checkpoint_path = os.path.join('./checkpoints', args.data, args.task)
    output_path = os.path.join('./output', args.data, args.task)

    train_acc = []
    valid_acc = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2) 
    train_begin = time.time()
    best_acc = 0
    for epoch in range(num_epochs):

    train(model, train_loader, valid_loader, lr=args.lr, batch_size=batch_size, num_epochs=args.epoch, print_every=args.print, checkpoint_path=checkpoint_path, output_path=output_path)
    test(model, test_loader, output_path, args.data)

    
