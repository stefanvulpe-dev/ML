import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
])

root_dir = 'C:\\Users\\stefa\\Documents\\ML\\neural_networks\\data\\cats_dogs'
tensor_dir = 'C:\\Users\\stefa\\Documents\\ML\\neural_networks\\data\\cats_dogs\\tensors\\train'
test_tensor_dir = 'C:\\Users\\stefa\\Documents\\ML\\neural_networks\\data\\cats_dogs\\tensors\\test'

os.makedirs(tensor_dir, exist_ok=True)
os.makedirs(test_tensor_dir, exist_ok=True)

files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

random.shuffle(files)

for counter, file in enumerate(files):
    if file.endswith('.jpg'):
        img = transform(Image.open(file).resize((300, 300)))
        file_name = os.path.splitext(os.path.basename(file))[0]
        if counter < (0.8 * 25000):
            torch.save(img, os.path.join(tensor_dir, f'{file_name}.pt'))
        else:
            torch.save(img, os.path.join(test_tensor_dir, f'{file_name}.pt'))


print('Done')
