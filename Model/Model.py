import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import os
import matplotlib.pyplot as plt
import random

from Dataset import Dataset
from Network import Alexnet

class Classification_Model():

    def __init__(self, _Net, _Classes_set, _data_loader, _device, _criterion, _optimizer, _schduler, _num_epoches=50):
        self.model = _Net
        self.classes_set = _Classes_set
        self.class_num = len(_Classes_set)
        self.data_loader = _data_loader
        self.device = _device
        self.criterion = _criterion
        self.optimizer = _optimizer
        self.schduler = _schduler
        self.num_epoches = _num_epoches

    def check_data(self, phase):
        dataset = self.data_loader[phase]
        idx = random.randint(0, len(dataset))
        sample = dataset.dataset[idx]
        print(idx, sample['image'].shape, self.classes_set[sample['label']])
        image = transforms.ToPILImage()(sample['image'])
        plt.imshow(image)
        plt.show()
        plt.close('all')

    def visualize(self):
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(self.data_loader['val']):
                inputs = data['image']
                labels = data['label'].to(self.device)

                x_classes = self.model(inputs.to(self.device)).view(-1, self.class_num)
                _, pred_label = torch.max(x_classes, 1)

                print(inputs.shape)
                plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
                plt.title('predicted classes: {}\n ground-truth label:{}'.format(self.classes_set[pred_label], self.classes_set[labels]))

    def train(self):
        self.Loss_list = {'train':[], 'val':[]}
        self.Acc_list = {'train':[], 'val':[]}
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.num_epoches):
            print('Epoch {}/{}'.format(epoch, self.num_epoches - 1))
            print('-*' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                correct_label = 0

                for idx, data in enumerate(self.data_loader[phase]):
                    print(phase + ' processing: {}th batch.'.format(idx))
                    inputs = data['image'].to(self.device)
                    labels = data['label'].to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred_label = self.model(inputs).view(-1, self.class_num)
                        _, pred_classes = torch.max(pred_label, 1)
                        loss = self.criterion(pred_label, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    correct_label += torch.sum(pred_classes == labels)

                epoch_loss = running_loss / len(self.data_loader[phase].dataset)
                self.Loss_list[phase].append(epoch_loss)
                epoch_correct = correct_label.double() / len(self.data_loader[phase].dataset)
                self.Acc_list[phase].append(100 * epoch_correct)
                print('{} Loss: {:.4f}  Accuracy: {:.2%}'.format(phase, epoch_loss, epoch_correct))

                if phase == 'val' and epoch_correct > best_acc:
                    best_acc = epoch_correct
                    best_model_wts = copy.deepcopy(self.model.state_dice())
                    print('Best val Accuracy: {:.2f}'.format(best_acc))

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), 'best_model.pt')
        print('Best val Accuracy: {:.2f}'.format(best_acc))

    def plot_loss(self):
        x = range(self.num_epoches)
        y1 = self.Loss_list['val']
        y2 = self.Loss_list['train']

        plt.plot(x, y1, color='r', linestyle='-', marker='o', linewidth=1, label='val')
        plt.plot(x, y2, color='b', linestyle='-', marker='o', linewidht=1, label='train')
        plt.legend()
        plt.title('train and val loss vs. epoches')
        plt.ylabel('loss')
        plt.savefig('train and val loss vs epoches.jpg')
        plt.close('all')

    def plot_acc(self):
        x = range(self.num_epoches)
        y1 = self.Acc_list['val']
        y2 = self.Acc_list['train']

        plt.plot(x, y1, color='r', linestyle='-', marker='.', linewidth=1, label='val')
        plt.plot(x, y2, color='b', linestyle='-', marker='.', linewidth=1, label='train')
        plt.legend()
        plt.title('train and val acc vs .epoches')
        plt.ylabel('acc')
        plt.savefig('train and val acc vs epoches.jpg')
        plt.close('all')

if __name__ == '__main__':
    sample_path = r'D:\Lynn\code\ChickenRatRabbit\Data'
    train_transforms = transforms.Compose([transforms.Resize((227, 227)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()
                                        ])
    val_transforms = transforms.Compose([transforms.Resize((227, 227)),
                                        transforms.ToTensor()
                                        ])
    train_Dataset = Dataset(os.path.join(sample_path, 'ChickenRatRabbit_train.csv'), train_transforms)
    val_Dataset = Dataset(os.path.join(sample_path, 'ChickenRatRabbit_val.csv'), val_transforms)
    
    train_dataloader = DataLoader(dataset=train_Dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(dataset=val_Dataset)
    data_loader = {'train': train_dataloader, 'val': val_dataloader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    network = Alexnet(3).to(device)
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    exp_lr_schduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    Sample_model = Classification_Model(network, ['Chicken', 'Rat', 'Rabbit'], data_loader, device, criterion, optimizer, exp_lr_schduler, 100)
    Sample_model.check_data('train')
    Sample_model.check_data('val')
    Sample_model.train()
    Sample_model.plot_acc()
    Sample_model.plot_loss
    Sample_model.visualize()




