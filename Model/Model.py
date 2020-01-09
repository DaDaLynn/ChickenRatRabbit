class Model():
    def __init__(self, _Net, _data_loader, _criterion, _optimizer, _schduler, _num_epoches=50):
        self.model = _Net
        self.data_loader = _data_loader
        self.criterion = _criterion
        self.optimizer = _optimizer
        self.schduler = _schduler
        self.num_epoches = _num_epoches

    def train(self):

        for epoch in range(self.num_epoches):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for _, data in enumerate(self.data_loader[phase])
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = 
