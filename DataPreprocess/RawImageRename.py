import os

class RawImageRename:
    def __init__(self, root, stage, species):
        self.DataPath = os.path.join(root, stage, species)
        self.species = species

    def rename(self):
        filenames = os.listdir(self.DataPath)
        n = 0
        for item in filenames:
            if item.endswith('.jpg'):
                oldname = os.path.join(self.DataPath, item)
                newname = os.path.join(self.DataPath, self.species + format(str(n), '0>3s') + '.jpg')
                os.rename(oldname, newname)
                n = n + 1
                print("{} th image is renamed".format(n))

if __name__ == "__main__":
    root = r"D:\Lynn\code\ChickenRatRabbit\Data"
    stage = ['train', 'val']
    species = ['chickens', 'rats', 'rabbits']
    
    for s in stage:
        for sp in species:
            mImageRename = RawImageRename(root, s, sp)
            mImageRename.rename()

