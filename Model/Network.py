'''
定义网络结构
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from Dataset import Dataset
import os
import torch

class Alexnet(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels= 384, padding=1, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.layer8 = nn.Linear(in_features=4096, out_features=num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 6*6*256)
        x = self.layer8(self.layer7(self.layer6(x)))
        return self.softmax(x)

if __name__ == '__main__':
    val_transforms = transforms.Compose([transforms.Resize((227, 227)),
                                        transforms.ToTensor()
                                        ])
    current_path = os.path.dirname(os.path.realpath(__file__))
    sample_path = os.path.abspath(current_path + os.path.sep + "../Data")
    nDataset = Dataset(os.path.join(sample_path, 'ChickenRatRabbit_val.csv'), val_transforms)
    nSample = nDataset[0]
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    model = Alexnet()
    print(model)
    with torch.set_grad_enabled(False):
        y_hat = model(nSample['image'].to(device).unsqueeze(0))
        _, pred_label = torch.max(y_hat.view(-1, 3), 1)
        print("True label is: {} Predict label is: {}".format(nSample['label'], pred_label.item()))
    for parameters in model.parameters():
        print(parameters)