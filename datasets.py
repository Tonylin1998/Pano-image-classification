import json
import pickle
import numpy as np
import pandas as pd
import torch
import os 
import sys
import math
import glob
import csv
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from os import listdir
import re
from datetime import datetime


class PanoDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames = []
        self.labels = []
        self.transform = transform
        pd_data1 = pd.read_csv(os.path.join(data_dir, 'data1.csv'),  encoding = "ISO-8859-1")
        pd_data2 = pd.read_csv(os.path.join(data_dir, 'data2.csv'),  encoding = "ISO-8859-1")
        pd_data3 = pd.read_csv(os.path.join(data_dir, 'data3.csv'),  encoding = "ISO-8859-1")

        ids = pd_data1['id'].to_numpy()[1:]
        dates = pd_data1['date'].to_numpy()[1:]

        label_a1 = pd_data1['Label A1'].to_numpy()[1:]
        label_b1 = pd_data1['Label B1'].to_numpy()[1:]

        assert(len(ids) == len(dates))
        for i, id, date in zip((range(len(ids))), ids, dates):
            #print(i, type(date), date)
            if(isinstance(date, str)):
                date = parser_date(date)
                possible_img = glob.glob(os.path.join(data_dir, '{}_{}_*.jpg'.format(id, date)))
                for img_path in possible_img:
                    self.filenames.append(img_path)

                    if(label_a1[i] == 'Y'):
                        a1 = 1
                    else:
                        a1 = 0

                    if(label_b1[i] == 'Y'):
                        b1 = 1
                    else:
                        b1 = 0
                    self.labels.append([a1, b1])

                    #img = Image.open(img_path)
                    #img = np.array(img)
                    #print(img.shape)

        ids = pd_data2['id'].to_numpy()[1:]
        dates = pd_data2['date'].to_numpy()[1:]

        label_a1 = pd_data2['Label A1'].to_numpy()[1:]
        label_b1 = pd_data2['Label B1'].to_numpy()[1:]

        assert(len(ids) == len(dates))
        for i, id, date in zip((range(len(ids))), ids, dates):
            #print(i, type(date), date)
            if(isinstance(date, str)):
                date = parser_date(date)
                possible_img = glob.glob(os.path.join(data_dir, '{}_{}_*.jpg'.format(id, date)))
                for img_path in possible_img:
                    self.filenames.append(img_path)

                    if(label_a1[i] == 'Y'):
                        a1 = 1
                    else:
                        a1 = 0

                    if(label_b1[i] == 'Y'):
                        b1 = 1
                    else:
                        b1 = 0
                    self.labels.append([a1, b1])



        filename = pd_data3['filename'].to_numpy()[1:]

        label_a1 = pd_data3['Label A1'].to_numpy()[1:]
        label_b1 = pd_data3['Label B1'].to_numpy()[1:]

       
        for i in range(len(filename)):
            #print(i, type(date), date)
            if(isinstance(filename[i], str)):
                self.filenames.append(os.path.join(data_dir, str(filename[i])+'.jpg'))

                if(label_a1[i] == 'Y'):
                    a1 = 1
                else:
                    a1 = 0

                if(label_b1[i] == 'Y'):
                    b1 = 1
                else:
                    b1 = 0
                self.labels.append([a1, b1])
        print('lenlen', len(self.labels))
      
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        label = torch.FloatTensor(self.labels[idx])

        image = Image.open(img_path).convert('L')
        if(self.transform != None):
            image = self.transform(image)

        return image, label
        
      
    def __len__(self):
        return len(self.filenames)

    
            
def parser_date(input_date):
    date1 = input_date.split('-')
    date2 = input_date.split('/')
    date3 = re.findall(r'\d+', input_date)
    if(len(date1) == 3):
        m = int(date3[0])
        d = int(date3[1])
        y = int(date3[2])
        date_str = '{:04}{:02}{:02}'.format(y,m,d)
    elif(len(date2) == 3):
        y = int(date3[0])
        m = int(date3[1])
        d = int(date3[2])
        date_str = '{:04}{:02}{:02}'.format(y,m,d)
    else:
        y = int(date3[0])
        m = int(date3[1])
        d = int(date3[2])
        date_str = '{:04}{:02}{:02}'.format(y,m,d)
    return date_str





if __name__ == '__main__':
    '''
    data_path = sys.argv[1]
    pd_data = pd.read_csv(os.path.join(data_path, 'info.csv'))

    ids = pd_data['id'].to_numpy()[1:]
    dates = pd_data['date'].to_numpy()[1:]

    print(ids)
    print(dates)

    assert(len(ids) == len(dates))
    for i, id, date in zip((range(len(ids))), ids, dates):
        print(i, type(date), date)
        if(isinstance(date, str)):
            date = parser_date(date)
            possible_img = glob.glob(os.path.join(data_path, '{}_{}_*.jpg'.format(id, date)))
            for img_path in possible_img:
                img = Image.open(img_path)
                img = np.array(img)
                print(img.shape)
    '''
    
    dataset = PanoDataset(sys.argv[1], 
                          transform=transforms.Compose([
                            transforms.CenterCrop(1024),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5], std=[0.5])
                            ]),
                        )

    dataloader = DataLoader(dataset = dataset,
                          batch_size = 16,
                          shuffle = False
                          )

    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels)






    '''
    usps_train_size = int(0.9 * len(usps_dataset))
    usps_valid_size = len(usps_dataset) - usps_train_size
    usps_train_dataset, usps_valid_dataset = torch.utils.data.random_split(usps_dataset, [usps_train_size, usps_valid_size])

    svhn_train_size = int(0.9 * len(svhn_dataset))
    svhn_valid_size = len(svhn_dataset) - svhn_train_size
    svhn_train_dataset, svhn_valid_dataset = torch.utils.data.random_split(svhn_dataset, [svhn_train_size, svhn_valid_size])

    mnistm_train_size = int(0.9 * len(mnistm_dataset))
    mnistm_valid_size = len(mnistm_dataset) - mnistm_train_size
    mnistm_train_dataset, mnistm_valid_dataset = torch.utils.data.random_split(mnistm_dataset, [mnistm_train_size, mnistm_valid_size])
    '''

