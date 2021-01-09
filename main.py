import numpy as np
import sys
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from datasets import PanoDataset
from models import ConvNet




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--data_dir', type=str, default='/tmp2/b06902028/multi_label/pano_data/', help='data dir')

    parser.add_argument('--model', type=str, help='model checkpoint path')
    parser.add_argument('--output', type=str, help='output filename')
    
    parser.add_argument('--gpu', type=int, default='0', help='gpu')


    args = parser.parse_args()
    # if(not args.train && not args.test):
    #   sys.exit('please choose at least one of train and test')

   
    return args

def f2_score(y_pred, y_true):
    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
 
    epsilon = 1e-10
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f2 = 5 * (precision*recall) / (4*precision + recall + epsilon)
    return f2

def train(epochs, dataset, dataloader, model, criterion, optimizer, scheduler):
    checkpoint_path = './model'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log = open('log.txt', 'w')

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_loss = 0
        train_acc = 0
        model.train()
        for i, data in enumerate(dataloader['train']):
            start_time = time.time()
            #print(episode)
            images = data[0].cuda()
            labels = data[1].cuda()
            
            optimizer.zero_grad()
            preds = model(images)
            #print(out.size())
            
            loss = criterion(preds, labels)
                        
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(preds)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            acc = f2_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
            train_loss += loss.item()*len(images)
            train_acc += acc*len(images)
    
            print('[{:05}/{:05}] loss: {:3.6f}, acc: {}'.format(i, len(dataloader['train']), loss, acc), end='\r')
        train_loss /= dataset['train'].__len__()
        train_acc /= dataset['train'].__len__()


        val_loss, val_acc = valid(dataset, dataloader, model, criterion, optimizer, scheduler)
        print('[{:05}/{:05}] {:2.2f} sec(s), train_loss: {:3.6f}, train_acc: {:3.6f}, val_loss: {:3.6f}, val_acc: {:3.6f}'.format(epoch, \
         epochs, time.time()-epoch_start_time, train_loss, train_acc, val_loss, val_acc))
        torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_{}.pkl'.format(epoch)))
        # if((episode) % 1000 == 0):
        #     val_loss, val_acc = valid(5, k_shot, n_query, n_episode, dis_mode, dis_func, dataloader, model, criterion, optimizer, scheduler, parametric_model)
        #     print('[{:05}/{:05}], {:2.2f} sec(s), val_loss: {:3.6f}, val_acc: {}'.format(episode+1, n_episode, time.time()-start_time, val_loss, val_acc))
        #     log.write('[{:05}/{:05}], {:2.2f} sec(s), val_loss: {:3.6f}, val_acc: {}\n'.format(episode+1, n_episode, time.time()-start_time, val_loss, val_acc))
        #     
        #     if(dis_mode == 'parametric'):
        #         torch.save(parametric_model.state_dict(), os.path.join(checkpoint_path, 'p_model_{}_{}.pkl'.format(dis_mode, episode)))
    log.close()


def valid(dataset, dataloader, model, criterion, optimizer, scheduler):   
    model.eval()  
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader['val']):

            images = data[0].cuda()
            labels = data[1].cuda()
            
            optimizer.zero_grad()
            preds = model(images)
            #print(out.size())
            
            loss = criterion(preds, labels)

          
            preds = torch.sigmoid(preds)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            acc = f2_score(preds, labels).cpu().detach().numpy()
        
             

            val_loss += loss.item()*len(images)
            val_acc += acc*len(images)
    val_loss /= dataset['val'].__len__()
    val_acc /= dataset['val'].__len__()

    return val_loss, val_acc
    
        
        


def main():
    args = parse_args()
    #print(args)
    torch.cuda.set_device(args.gpu)
    epochs=20

    model = ConvNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    #print(model)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=6, min_lr=1e-6, verbose=True)


    dataset = {}
    total_dataset = PanoDataset(args.data_dir, 
                          transform=transforms.Compose([
                            transforms.CenterCrop(1024),
                            transforms.Resize(256),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5], std=[0.5])
                            ]),
                        )
    train_size = int(0.9 * len(total_dataset))
    valid_size = len(total_dataset) - train_size
    dataset['train'], dataset['val'] = torch.utils.data.random_split(total_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
    print('train set length: ', dataset['train'].__len__())
    print('val set length: ', dataset['val'].__len__())

    dataloader = {}
    dataloader['train'] = DataLoader(dataset = dataset['train'],
                          batch_size = 16,
                          shuffle = True
                          )
    dataloader['val'] = DataLoader(dataset = dataset['val'],
                          batch_size = 16,
                          shuffle = False
                          )
    '''
    dataiter = iter(dataloader['train'])
    images, labels = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels)

    out = model(images.cuda())
    print(out.size())
    '''

    train(epochs, dataset, dataloader, model, criterion, optimizer, scheduler)





if __name__ == '__main__':
    main()