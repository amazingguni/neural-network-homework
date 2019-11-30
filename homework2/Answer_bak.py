import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time

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
    def __init__(self, x_path, y_path, data, transform=None):
        self.x_data = np.load(x_path)
        self.has_label = y_path != None
        self.y_data = np.argmax(np.load(y_path), axis=1) if y_path else None
        self.x_shape = self.x_data.shape
        self.y_shape = self.y_data.shape if y_path else None
        self.len = self.x_shape[0]
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        image = self.x_data[index]
        if self.data == 'cifar':
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)
            image = Image.fromarray(image, 'RGB')
        elif self.data == 'mnist':
            image = image.reshape(28, 28)
            image = Image.fromarray(np.uint8(image * 255) , 'L')
            image = image.convert(mode='RGB')
        if self.transform:
            image = self.transform(image)
        label = self.y_data[index] if self.has_label else None
        return image, label

    def __len__(self):
        return self.len


def load_dataset(data, task, transform_train, transform_test, batch_size=100):
    # check the data & task you selected
    print('#'*50)
    print('DATA: {}'.format(data))
    print('TASK: {}'.format(task))
    print('#'*50)

    # Load the dataset
    data_path = os.path.join('./data', data, task)
    train_dataset = HomeworkDataset(os.path.join(data_path, 'train_x.npy'), os.path.join(data_path, 'train_y.npy'), data, transform_train)
    #train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, sampler=ImbalancedDatasetSampler(train_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = HomeworkDataset(os.path.join(data_path, 'valid_x.npy'), os.path.join(data_path, 'valid_y.npy'), data, transform_test)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = HomeworkDataset(os.path.join(data_path, 'test_x.npy'), None, data, transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    # Check the shape of your data
    print(f'x_train shape: {train_dataset[0][0].shape}')
    print(f'x_valid shape: {valid_dataset[0][0].shape}')
    print(f'x_test shape: {test_dataset[0][0].shape}')
    print('#'*50)

    return train_loader, valid_loader, test_loader


#########################################################################################################
# ----------------------------------------- SAMPLE CODE ------------------------------------#
# No need to use the code below.
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def train(model, train_loader, valid_loader, batch_size, lr, num_epochs, print_every=5, checkpoint_path=None):
    train_acc = []
    valid_acc = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2) 
    train_begin = time.time()
    #model.train()
    best_acc = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_begin = time.time()
        for batch_data in train_loader:    
            images, labels = batch_data 
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            pred = outputs.argmax(dim=1)
            #correct += torch.sum(pred.eq(labels.argmax(dim=1))).item()
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

def test(model, test_loader, save_path, data):
    model.eval()
    test_data = next(iter(valid_loader))
    images, _ = test_data
    images = Variable(images).to(device)
    test_pred = model(images)
    test_pred = np.argmax(test_pred.data.cpu().numpy(), axis=1)
    true = np.load(f'{data}_test_y.npy')
    total = test_pred.shape[0]
    correct = len(np.where(test_pred == true)[0])
    real_acc = correct / total
    print(','.join([f'({pred}, {label})' for pred, label in zip(test_pred, true)]))
    print(f'Real Accuracy = {real_acc * 100:.4f}({correct}/{total})')

    # You need to save and submit the 'test_pred.npy' under the output directory.
    np.save(os.path.join(save_path, 'test_pred.npy'), test_pred)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=200, help='num_epochs')
    parser.add_argument('--data', default='mnist', help='mnist, cifar')
    parser.add_argument('--task', default='1_imbalanced', help='1_imbalanced, 2_semisupervised, 3_noisy')
    parser.add_argument('--print', type=int, default=10, help='Print every..')
    
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # https://github.com/kuangliu/pytorch-cifar/issues/19
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_loader, valid_loader, test_loader = load_dataset(args.data, args.task, transform_train, transform_test, args.batch)
    
    trainset = torchvision.datasets.CIFAR100(root='./data_orig', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
    train_cifar100 = trainset[0]
    print(f'cifar100 origin size: {train_cifar100[0].shape}')
    testset = torchvision.datasets.CIFAR100(root='./data_orig', train=False, download=True, transform=transform_test)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)

    #model = models.resnet18(pretrained=True).to(device)
    model = models.resnet18().to(device)
    if device == 'cuda':
        print('Turn on DataParallel')
        model = torch.nn.DataParallel(model)
        
    checkpoint_path = os.path.join('./checkpoints', args.data, args.task)
    train(model, train_loader, valid_loader, lr=args.lr, batch_size=args.batch, num_epochs=args.epoch, print_every=args.print, checkpoint_path=checkpoint_path)
    save_path = os.path.join('./output', args.data, args.task)
    test(model, test_loader, save_path, args.data)

    
