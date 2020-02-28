import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import albumentations as A

scene_dict_file = open('scene_dict.json','r')
scene_dict = json.load(scene_dict_file)
s_by_i = {int(i):key for i,key in scene_dict[0].items()}
i_by_s = {key:int(i) for key,i in scene_dict[1].items()}
train_folder = 'drivedata/train/'
test_folder = 'drivedata/test_scene/'
print(s_by_i)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
print(torch.cuda.get_device_name(0))
class SceneDataset(Dataset):
    def __init__(self, folder, transform=None, augmentation=None):
        self.augmentation = augmentation
        self.transform = transform
        self.folder = folder
        filenames = os.listdir(self.folder)
        self.filenames = [name for name in filenames if name != 'desktop.ini' and name.split('_')[1]!='undefined']
        splited = [x.replace('.jpg','').split('_') for x in self.filenames]
        self.splited = [[i_by_s[x[1]],x[0]] for x in splited]
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.folder, self.filenames[index]))
        y = self.splited[index][0]
        img_id = self.splited[index][1]
        if self.augmentation:
            img = self.augmentation(image = np.array(img))['image']
        if self.transform: 
            img = self.transform(img)
        return img, y, img_id

def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):    
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train() # Enter train mode
        
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        start = time.time()
        for i_step, (x, y,_) in enumerate(train_loader):
            start = time.time()
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            
            prediction = model(x_gpu)    
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]
            
            loss_accum += loss_value
            print(i_step,  time.time()- start)
        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)
        name = 'scene_'+time.strftime('%Y-%m-%d_%H-%M_')+str(epoch)+'_'+str(val_accuracy)+'.pt'
        PATH = os.path.join('models_checkpoint/',name)
        model.cpu()
        torch.save(model, PATH)
        model.to(device)
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        
        print("Average loss: %f, Train accuracy: %f, Val balanced accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))
        
    return loss_history, train_history, val_history
        
def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader
    
    Returns: accuracy as a float value between 0 and 1
    """
    model.eval() # Evaluation mode
    pred = list()
    gr_tr = list()
    for (x,y,_) in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        predict = model(x)
        x.cpu()
        _, indices = torch.max(predict, 1)
        gr_tr.extend(y.cpu().tolist())
        pred.extend(indices.cpu().tolist())
        
    val_accuracy = metrics.balanced_accuracy_score(gr_tr, pred)
    with open('val_result.json', 'w') as outfile:
        json.dump([pred,gr_tr], outfile)
    return val_accuracy

def init_aug():
    augmentation_pipeline = A.Compose(
    [
        A.ShiftScaleRotate(rotate_limit=10),
        A.HorizontalFlip(p = 0.5),
        A.OneOf(
            [
                A.RandomContrast(), 
                A.RandomGamma(), 
                A.RandomBrightness(),
            ],
            p = 0.5
        ),
        
    ],
    p = 1
    )
    return augmentation_pipeline

def main():
    orig_dataset = SceneDataset('drivedata/train/')
    train_dataset = SceneDataset(train_folder, 
                           transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               # Use mean and std for pretrained models
                               # https://pytorch.org/docs/stable/torchvision/models.html
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])                         
                           ]),
                       augmentation=init_aug()
                          )
    test_dataset =SceneDataset(test_folder, 
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               # Use mean and std for pretrained models
                               # https://pytorch.org/docs/stable/torchvision/models.html
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])                         
                           ])
                          )

    batch_size = 16
    data_size = len(orig_dataset)
    validation_fraction = .1
    val_split = int(np.floor((validation_fraction) * data_size))
    indices = list(range(data_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    indices = indices
    val_indices, train_indices = indices[:val_split], indices[val_split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=True,batch_size=batch_size, 
                                               sampler=train_sampler,num_workers=16)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             sampler=val_sampler,pin_memory=True,num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,pin_memory=True,num_workers=16)
    model = torch.load('models_checkpoint/scene_2020-02-23_06-01_3_0.8644037972165372.pt')
    model.eval()
    #model = models.resnext50_32x4d(pretrained=True)
    #num_ftrs = model.fc
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 8)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': model.fc.parameters(), 'lr': 1e-4},
                        {'params': list(model.parameters())[:-2]}],
                         lr=0.00001, momentum=0.9)
    loss_history, train_history, val_history = train_model(model, train_loader, val_loader, loss, optimizer, 5)
if __name__ == '__main__':
    main()