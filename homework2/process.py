import os
import numpy as np
from PIL import Image

data_dirs = [
    'cifar',
    'mnist',
]

task_dirs = [
    '1_imbalanced',
    '2_semisupervised',
    '3_noisy',
]

def process(data, task, name, has_label=True):
    data_path = os.path.join('./data', data, task)

    x_data = np.load(os.path.join(data_path, name + '_x.npy'))
    if has_label:
        y_data = np.load(os.path.join(data_path, name + '_y.npy'))
        label = np.argmax(y_data, axis=1)

    if data == 'cifar':
        images = x_data.reshape(-1, 3, 32, 32)
    else:
        images = x_data.reshape(-1, 28, 28)

    for i, image in enumerate(images):
        if data == 'cifar':
            img = Image.fromarray(np.uint8(image * 255).transpose(1, 2, 0), 'RGB')
        else:
            img = Image.fromarray(np.uint8(image * 255) , 'L')
        if has_label:
            image_path = os.path.join('./images', data, task, name, str(label[i]))            
        else:
            image_path = os.path.join('./images', data, task, name)
        os.makedirs(image_path, exist_ok=True)
        img.save(os.path.join(image_path, str(i) + '.png'))

for data in data_dirs:
    for task in task_dirs:
        process(data, task, 'train')
        process(data, task, 'valid')
        process(data, task, 'test', has_label=False)

