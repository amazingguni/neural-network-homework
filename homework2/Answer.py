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
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn.init import xavier_uniform_
import logging
from datetime import datetime
from models.resnet import resnet18, resnet34
import torch.nn.functional as F 
from loss import SCELoss
import wandb

wandb.init(project="homework2")

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

stream_hander = logging.StreamHandler()
logger.addHandler(stream_hander)



torch.manual_seed(1021)
torch.cuda.manual_seed(1021)
np.random.seed(1021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    logger.info(f'cuda is available: {visible_devices}')
    torch.backends.cudnn.benchmark = True

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
        if self.transform:
            image = self.transform(image)
        label = np.int(-1)
        if self.y_data is not None:
            label = self.y_data[index]
        return image, label

    def __len__(self):
        return self.len

def draw(train_acc, test_acc, save_path):
    x_axis = list(range(len(test_acc)))
    plt.plot(x_axis, train_acc, 'b-', label='Train Acc.')
    plt.plot(x_axis, test_acc, 'r-', label='Test Acc.')
    plt.title('Train & Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.clf()

def train(model, train_loader, criterion, optimizer):
    epoch_loss = 0.0
    correct = 0
    total = 0
    model.train()
    current_self_faced_warm_step = 0
    for batch_data in train_loader:
        images, labels = batch_data 
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += torch.sum(pred.eq(labels)).item()
        total += outputs.shape[0]
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= total
    return epoch_loss, correct / total


def validate(model, valid_loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += torch.sum(pred.eq(labels)).item()
        total += outputs.shape[0]
    return correct / total

def test(model, test_loader):
    model.eval()
    total_pred = np.array([])
    for images, _ in test_loader:
        images = Variable(images).to(device)
        test_pred = model(images)
        test_pred = np.argmax(test_pred.data.cpu().numpy(), axis=1)
        total_pred = np.hstack([total_pred, test_pred])
    return total_pred


def make_balanced_classes(x_data, y_data, num_classes):
    argmax_y_data = np.argmax(y_data, axis=1)
    u, counts = np.unique(argmax_y_data, return_counts=True)
    max_count = max(counts)
    min_count = min(counts)
    if max_count == min_count:
        return x_data, y_data
    num_of_data_for_fewer_classes = int((max_count / min_count)) - 1
    for idx, count in zip(u, counts):
        if count != min_count:
            continue
        imbalanced_label_indexes = np.where(argmax_y_data == idx)
        x_data = np.vstack((x_data, np.tile(x_data[imbalanced_label_indexes], (num_of_data_for_fewer_classes, 1))))
        y_data = np.vstack((y_data, np.tile(y_data[imbalanced_label_indexes], (num_of_data_for_fewer_classes, 1))))
    return x_data, y_data
    

def split_unlabeled_dataset(x_data, y_data):
    exists = y_data.sum(axis=1)
    u, counts = np.unique(exists, return_counts=True)
    labeled_x = x_data[np.where(exists==1)]
    labeled_y = y_data[np.where(exists==1)]
    unlabeled_x = x_data[np.where(exists==0)]
    
    logger.info('# split_unlabeled_dataset #')
    logger.info(f'  * labeled_data_shape: {labeled_x.shape} {labeled_y.shape}')
    logger.info(f'  * unlabeled_data_shape: {unlabeled_x.shape}')
    
    return labeled_x, labeled_y, unlabeled_x 

def print_args(args, output_path):
    logger.info('#'*50)
    logger.info(f'output_path: {output_path}')
    logger.info('DATA: {}'.format(args.data))
    logger.info('TASK: {}'.format(args.task))
    logger.info(f'use all: {args.use_all}')
    logger.info(f'epoch: {args.epoch}')
    logger.info(f'batch_size: {args.batch_size}')
    logger.info(f'only_valid: {args.only_valid}')
    if args.loss == 'sce':
        logger.info(f'loss: {args.loss}(a:{args.sce_alpha}, b:{args.sce_beta})')
    else:
        logger.info(f'loss: {args.loss}')
    if args.task == '1_imbalanced':
        logger.info(f'balanced: {args.balanced}')        
    if args.task == '3_noisy':
        logger.info(f'revise_label: {args.revise_label}')
        logger.info(f'sub_ckpt: {args.sub_ckpt}')
        logger.info(f'confidence: {args.confidence}')
    if args.semi:
        logger.info(f'semi: {args.semi}')
        logger.info(f'sub_ckpt: {args.sub_ckpt}')
        logger.info(f'confidence: {args.confidence}')
    logger.info('#'*50)

if __name__ == '__main__':    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=200, help='num_epochs')
    parser.add_argument('--data', default='mnist', help='mnist, cifar')
    parser.add_argument('--task', default='1_imbalanced', help='1_imbalanced, 2_semisupervised, 3_noisy, origin')
    parser.add_argument('--revise-label', action='store_true', help='True if you want to revise noisy label')
    parser.add_argument('--loss', default='ce', help='ce, sce')
    parser.add_argument('--sce-alpha', type=float, default=0.1, help='alpha value for SCE, it is multiplied with CE in SCE')
    parser.add_argument('--sce-beta', type=float, default=1.0, help='beta value for SCE')
    parser.add_argument('--print', type=int, default=10, help='Print every..')
    parser.add_argument('--use-all', action='store_true', help='True if you want to use both valid and train for training')
    parser.add_argument('--balanced', action='store_true', help='True if you want to balance data set')
    parser.add_argument('--semi', action='store_true', help='True if you want to label unlabeled data')
    parser.add_argument('--sub-ckpt', type=str, help='The path of weight for labeling or revising')
    parser.add_argument('--confidence', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--only-valid', action='store_true', help='True, if you want to train with only validation set.')

    
    args = parser.parse_args()

    data = args.data
    task = args.task

    num_classes = 10 if data == 'mnist' else 100
    dir_name = datetime.now().strftime('%y%m%d-%H%M')
    if args.use_all:
        dir_name = dir_name + '-use-all'
    output_path = Path('./output') / data / task / dir_name
    output_path.mkdir(exist_ok=True)
    checkpoint_path = output_path / 'best.pth'
    test_label_path = output_path / 'test_pred.npy'
    plot_path = output_path / 'plot.png'
    log_path = output_path / 'training.log'
    
    logger.addHandler(logging.FileHandler(log_path))

    print_args(args, output_path)
    
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

        train_y_data = np.load(os.path.join(data_path, 'train_y.npy'))
        valid_y_data = np.load(os.path.join(data_path, 'valid_y.npy'))
        test_y_data = np.load(f'{data}_test_y.npy')
        test_y_data = np.eye(num_classes)[test_y_data]

        if args.only_valid:
            train_x_data = valid_x_data
            train_y_data = valid_y_data
            valid_x_data = test_x_data
            valid_y_data = test_y_data
            
        if args.balanced:
            train_x_data, train_y_data = make_balanced_classes(train_x_data, train_y_data, num_classes)
        if task == '2_semisupervised':
            train_x_data, train_y_data, unlabeled_x_data = split_unlabeled_dataset(train_x_data, train_y_data)
        if args.only_valid:
            train_x_data = np.concatenate((train_x_data, valid_x_data), axis=0)
            train_y_data = np.concatenate((train_y_data, valid_y_data), axis=0)
            valid_x_data = test_x_data
            valid_y_data = test_y_data
        train_y_data = np.argmax(train_y_data, axis=1)
        valid_y_data = np.argmax(valid_y_data, axis=1)
        if args.revise_label and args.sub_ckpt:
            model = resnet18(num_classes=num_classes).to(device)
            logger.info(f'Revise noisy label data: {args.sub_ckpt}')
            model.load_state_dict(torch.load(args.sub_ckpt))
            noisy_dataset = HomeworkDataset(train_x_data, None, data, transform_test)
            noisy_loader = DataLoader(dataset=noisy_dataset, batch_size=batch_size, shuffle=False)
            pred_labels = np.array([], dtype=int)
            confidences = np.array([], dtype=int)
            for batch_data in noisy_loader:
                images, _ = batch_data 
                images = Variable(images).to(device)
                outputs = model(images)
                values, indices  = F.softmax(outputs, dim=None).max(axis=1)
                pred_labels = np.hstack([pred_labels, indices.cpu()])
                confidences = np.hstack([confidences, values.detach().cpu()])
            train_y_data = np.where(confidences > args.confidence, pred_labels, train_y_data)
        if args.semi and args.sub_ckpt:
            model = resnet18(num_classes=num_classes).to(device)
            logger.info(f'Label unlabeled data: {args.sub_ckpt}')
            model.load_state_dict(torch.load(args.sub_ckpt))
            unlabeled_dataset = HomeworkDataset(unlabeled_x_data, None, data, transform_test)
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=batch_size, shuffle=False)
            pred_labels = np.array([], dtype=int)
            confidences = np.array([], dtype=int)
            for batch_data in unlabeled_loader:
                images, _ = batch_data 
                images = Variable(images).to(device)
                outputs = model(images)
                values, indices  = F.softmax(outputs, dim=None).max(axis=1)
                pred_labels = np.hstack([pred_labels, indices.cpu()])
                confidences = np.hstack([confidences, values.detach().cpu()])
            confidence_indice = np.where(confidences > args.confidence)
            labeled_x_data = unlabeled_x_data[confidence_indice]
            labeled_y_data = pred_labels[confidence_indice]
            logger.info(f'labeled_x_data: {labeled_x_data.shape}, labeled_y_data: {labeled_y_data.shape}')
            train_x_data = np.concatenate((train_x_data, labeled_x_data), axis=0)
            train_y_data = np.concatenate((train_y_data, labeled_y_data), axis=0)
            
        if args.use_all:
            train_x_data = np.concatenate((train_x_data, valid_x_data), axis=0)
            train_y_data = np.concatenate((train_y_data, valid_y_data), axis=0)
            valid_x_data = test_x_data
            valid_y_data = test_y_data
            valid_y_data = np.argmax(valid_y_data, axis=1)
        train_dataset = HomeworkDataset(train_x_data, train_y_data, data, transform_train)
        valid_dataset = HomeworkDataset(valid_x_data, valid_y_data, data, transform_test)
        test_dataset = HomeworkDataset(test_x_data, test_y_data, data, transform_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    

    # Check the shape of your data
    logger.info(f'x_train shape: {len(train_dataset)} {train_dataset[0][0].shape}')
    logger.info(f'x_valid shape: {len(valid_dataset)} {valid_dataset[0][0].shape}')
    logger.info(f'x_test shape: {len(test_dataset)} {test_dataset[0][0].shape}')
    logger.info('#'*50)


    train_acc = []
    valid_acc = []

    model = resnet18(num_classes=num_classes).to(device)
    wandb.watch(model)
    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
        logger.info(' - CrossEntropyLoss()')
    elif args.loss == 'sce':
        logger.info(f' - SCELoss(alpha={args.sce_alpha}, beta={args.sce_beta}')
        criterion = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2) 
    train_begin = time.time()
    best_acc = 0
    best_checkpoint = ''


    for epoch in range(args.epoch):
        epoch_begin = time.time()
        epoch_loss, epoch_train_acc = train(model, train_loader, criterion, optimizer)
        train_acc.append(epoch_train_acc)
        epoch_valid_acc = validate(model, valid_loader)
        valid_acc.append(epoch_valid_acc)
        if best_acc < epoch_valid_acc and checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)
            best_acc = epoch_valid_acc 
            test_pred = test(model, test_loader)
            np.save(test_label_path, test_pred)
            # need to remove

            true = np.load(f'{data}_test_y.npy')
            total = true.shape[0]
            correct = len(np.where(test_pred == true)[0])
            real_acc = correct / total
            wandb.log({"Real Accuracy": real_acc * 100})
            logger.info(f'Real Accuracy = {real_acc * 100:.4f}')
        
        if epoch % args.print == 0:
            current = time.time()
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 60.0
            wandb.log({
                "Train Accuracy": epoch_train_acc * 100,
                "Valid Accuracy": epoch_valid_acc * 100,
                "Train Loss": epoch_loss,
            })
            logger.info(f'[EPOCH {epoch}] Loss = {epoch_loss:.4f} elapsed: epoch({epoch_elapsed:.3f}m) total({train_elapsed:.3f}m)')
            logger.info(f'Train Accuracy = {epoch_train_acc * 100:.4f}')
            logger.info(f'Valid Accuracy = {epoch_valid_acc * 100:.4f}(best: {best_acc * 100:.4f})')
        train_scheduler.step(epoch)

    draw(train_acc, valid_acc, plot_path)

    print_args(args, output_path)
    logger.info(f'log path: {log_path}')
    logger.info(f'best valid acc: {best_acc}')
    logger.info(f'best test result: {test_label_path}')
    logger.info(f'best ckpt: {checkpoint_path}')


