import os

class RawImageRename:
    def __init__(self, _root, _stage, _species):
        self.DataPath = os.path.join(_root, _stage, _species)
        self.species = _species

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
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_root_path = os.path.abspath(current_path + os.path.sep + "../Data")
    stage = ['train', 'val']
    species = ['chickens', 'rats', 'rabbits']
    
    for s in stage:
        for sp in species:
            mImageRename = RawImageRename(data_root_path, s, sp)
            mImageRename.rename()

