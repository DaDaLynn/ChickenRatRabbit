'''
定义数据结构
'''
import os
import pandas as pd
from PIL import Image
class Dataset():
    def __init__(self, _anno_csv, _transform):
        if os.path.exists(_anno_csv):
            self.df = pd.read_csv(_anno_csv)
            self.transform = _transform
        else:
            raise Exception("{} do not exist, please check.".format(_anno_csv))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= 0 and idx < len(self.df):
            image_path = self.df['path'][idx]
            image_label = self.df['label'][idx]
            image_data = Image.open(image_path).convert('RGB')
            if self.transform:
                image_data = self.transform(image_data)
            return {'image': image_data, 'label': image_label}
        else:
            raise Exception("index out of data size")

if __name__ == '__main__':
    sample_path = r'D:\Lynn\code\ChickenRatRabbit\Data'
    nDataset = Dataset(os.path.join(sample_path, 'ChickenRatRabbit_val.csv'), None)
    nSample = nDataset[0]
