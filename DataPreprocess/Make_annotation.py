import pandas as pd
import os

class Make_annotation():
    def __init__(self, _path, _phase, _species):
        self.valid_data_folder_info = []
        species_label = list(zip(_species, range(len(_species))))
        for valid_phase in _phase:
            valid_data_folder = []
            for valid_species,valid_label in species_label:
                absolute_path = os.path.join(_path, valid_phase, valid_species)
                if os.path.exists(absolute_path):
                    valid_data_folder.append({'path':absolute_path, 'phase':valid_phase, 'label':valid_label})
                else:
                    raise Exception("{} do not exist".format(absolute_path))
            self.valid_data_folder_info.append({'phase': valid_phase, 'data': valid_data_folder})

    def make(self, csv_path, prefix):
        for valid_phase_folder in self.valid_data_folder_info:
            annotation = {'path':[], 'label':[]}
            _phase = valid_phase_folder['phase']
            for valid_species_folder in valid_phase_folder['data']:
                _path  = valid_species_folder['path']
                _label = valid_species_folder['label']
                file_list = os.listdir(_path)
                annotation['path'] = annotation['path'] + [os.path.join(_path, f) for f in file_list]
                annotation['label'] = annotation['label'] + [_label for i in range(len(file_list))]
            df_csv = pd.DataFrame.from_dict(annotation)
            df_csv.to_csv(os.path.join(csv_path, '_'.join([prefix, _phase]) + '.csv'))


if __name__ == '__main__':
    sample_path = r'D:\Lynn\code\ChickenRatRabbit\Data'
    phase = ['train', 'val']
    species = ['chickens', 'rabbits', 'rats']
    nSample = Make_annotation(sample_path, phase, species)
    nSample.make(sample_path, 'ChickenRatRabbit')