import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import copy

#########################

# GLOBAL VARIABLES
# directory with images folder
DATA_DIR = 'data'
# path to the model, in order to both saving and loading it
MODEL_PATH = 'model_1.pth'

# percantage of data to be validated
VAL_SIZE = 0.4

# decide if the files are to be moved to subfolders
MOVE_FILES = False # SET IT TO TRUE ONLY WITH THE FIRST RUN OF CODE!
# decide if the model is to be trained and saved or just loaded
TRAIN = False
# decide if the parameters of model are to be frozen or not
# in other words, decide if you want train the last layer or fine tune all layers
FINE_TUNE_ALL = False

#########################

train_labels = pd.read_csv(os.path.join(DATA_DIR,'train.csv'))
train_size = int((1-VAL_SIZE) * len(train_labels))
val_size = len(train_labels) - train_size

class_names = train_labels.columns[1:].values
class_num = len(class_names)
print(class_names)

# auxiliary function used for creating class folders and moving files into them
def moveFiles(labels): 
    
    # create folders for train, validation and test
    for x in ['train','val','test']:
        if os.path.isdir(os.path.join(DATA_DIR,x)) == False:
            os.mkdir(os.path.join(DATA_DIR,x))
    
    # move all test files to test folder
    for filename in os.listdir(os.path.join(DATA_DIR, 'images')):
        if filename.startswith('Test'):
            os.rename(os.path.join(os.getcwd(),DATA_DIR,'images',filename),
                      os.path.join(os.getcwd(),DATA_DIR,'test',filename))
            
    # create folders for every class in train and validation folders
    for x in ['train','val']:
        for class_name in class_names:
            if os.path.isdir(os.path.join(DATA_DIR,x,class_name)) == False:
                os.mkdir(os.path.join(DATA_DIR,x,class_name))
     
    # move every file into its class folder
    # train
    for index, row in labels[:train_size].iterrows():
        file_class = row[row.isin([1])].index.values[0]
        file_id = row['image_id'] + '.jpg'
        os.rename(os.path.join(os.getcwd(),DATA_DIR,'images',file_id),
                  os.path.join(os.getcwd(),DATA_DIR,'train',file_class,file_id))
    
    # val
    for index, row in labels[train_size:].iterrows():
        file_class = row[row.isin([1])].index.values[0]
        file_id = row['image_id'] + '.jpg'
        os.rename(os.path.join(os.getcwd(),DATA_DIR,'images',file_id),
                  os.path.join(os.getcwd(),DATA_DIR,'val',file_class,file_id))
        
                   
# do it only once!!!     
if MOVE_FILES:
    moveFiles(train_labels)


# data augmentation and normalization for training
# just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR,x),
                                          data_transforms[x]) 
                    for x in ['train','val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
                for x in ['train','val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
print(dataset_sizes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# visualize a few images in order to understand data augmentation
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# general function to train model
# scheduling the learning rate
# saving the best model

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    
    # initial values
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0

            # iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) 
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            if phase == 'train':
                scheduler.step()                    
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualizeModel(model, num_images=6):
    
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
                
        model.train(mode=was_training)                
                
# use pretrained version of resnet50
model = models.resnet50(pretrained=True)

# freeze model parameters
for param in model.parameters():
    param.requires_grad = FINE_TUNE_ALL

# number of input features in the fully connected layer
num_ftrs = model.fc.in_features

# replace the last layer with linear unit with 4 outputs
model.fc = nn.Linear(num_ftrs, class_num)

# calculate on cuda if available
model.to(device)

# because it is multi-class problem, therefore we use cross entropy loss
criterion = nn.CrossEntropyLoss()

if FINE_TUNE_ALL:
    # all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=0.001)
else:
    # only parameters of final layer are being optimized 
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

if TRAIN:
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=20)
    torch.save(model, MODEL_PATH)
else:
    model = torch.load(MODEL_PATH)
    model.eval()


visualizeModel(model)


