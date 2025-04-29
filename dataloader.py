import glob
from sklearn.model_selection import train_test_split
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.transform import rescale, resize, downscale_local_mean
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import utils

# training_dataset_total = datasets.ImageFolder(root=utils.dirs['train'], transform=utils.transform['train'])
# testing_dataset_total = datasets.ImageFolder(root=utils.dirs['test'], transform=utils.transform['test'])


class SilicosisDataset(Dataset):
    """Silicosis Disease dataset."""

    def __init__(self,filenames,mode, transform=None):
        """
        Args:
            filenames (list): list of filenames of images
            input_size (Integer) : size of the input
            transform (callable, optional): Optional transform to be applied
                on a sample.
            model (string) : Tells wether dataloader for train or test/val. 
        """
        self.filenames = filenames
        self.transform = transform
        self.count_normal = 0
        self.count_silicosis = 0
        self.count_stb = 0
        self.count_tb = 0
        self.mode = mode

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.filenames[idx]
        
        if 'folder_normal' in image_name:
            label = 0
            self.count_normal +=1
            
        elif 'folder_TB' in image_name:
            label = 1
            self.count_tb  +=1
           
        elif 'folder_silicosis' in image_name:
            label = 2
            self.count_silicosis +=1
            
        elif 'folder_STB' in image_name:
            label = 3
          
        image = io.imread(image_name)
        #image = image/255.0
        image = torch.from_numpy(image)
        #image = image/255.0
        image = image.permute(2, 0, 1)
        
        
        train_transforms = utils.transform['train']
        test_transforms = utils.transform['test']
        # transforms.Compose([transforms.ToPILImage(),
        #  transforms.ToTensor()])#transforms.Grayscale(),transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),  transforms.RandomRotation(45),transforms.RandomRotation(20),])#,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # test_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])#,transforms.Grayscale(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        if self.transform:
            if self.mode == 'train':
                image = train_transforms(image)
                
            else:
                image = test_transforms(image)
                


        #print(self.count_normal, self.count_silicosis, self.count_stb, self.count_tb)
        return image,label
    
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

files_total_train = glob.glob(utils.dirs['train']+'/**/*.jpg', recursive=True)
files_total_test = glob.glob(utils.dirs['test']+'/**/*.jpg', recursive=True)

print(len(files_total_test), len(files_total_train))


training_dataset_total = SilicosisDataset(files_total_train,'train',transform=True)
testing_dataset_total = SilicosisDataset(files_total_test,'test',transform=True)

