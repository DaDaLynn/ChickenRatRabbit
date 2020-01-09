from Dataset import Dataset
import matplotlib.pyplot as plt
import os

class Visualize_data():
    def __init__(self, dataset):
        self.dataset = dataset

    def visualize(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        plt.imshow(image)
        plt.show()

if __name__ == '__main__':
    sample_path = r'D:\Lynn\code\ChickenRatRabbit\Data'
    nDataset = Dataset(os.path.join(sample_path, 'ChickenRatRabbit_val.csv'), None)
    Visual = Visualize_data(nDataset)
    Visual.visualize(0)