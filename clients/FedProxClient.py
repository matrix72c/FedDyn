from interface import client
import torch
import torch.utils.data as Data
from model import NCFModel
from utils import *
from torch.optim.lr_scheduler import StepLR
from torch import nn

class FedProxClient(client.Client):
    def __init__(self, train_data, train_label, user_num, item_num, run, client_id=0):
        self.model = NCFModel(user_num, item_num, run['hparams']['factor_num'], run['hparams']['num_layers'], run['hparams']['dropout'], run['hparams']['model_type'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=run['hparams']['learning_rate'])
        self.schedule = StepLR(self.optimizer, step_size=run['hparams']['lr_step'], gamma=run['hparams']['lr_decay'])
        self.dataset = NCFDataset(torch.tensor(train_data).to(torch.long), torch.tensor
        (train_label).to(torch.float32))
        self.loader = Data.DataLoader(self.dataset, batch_size=run['hparams']['batch_size'], shuffle=True, num_workers=0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = run['hparams']['epochs']
        super(FedProxClient, self).__init__(train_data, train_label, user_num, item_num, run, client_id)
    
    def train(self):
        server_params = self.model.parameters()
        model = self.model.to(self.device)
        model.train()
        for epoch in range(self.epochs):
            for data in self.loader:
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                y_ = self.model(x)
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), server_params):
                    proximal_term += (w - w_t).norm(2)
                loss = nn.BCEWithLogitsLoss()(y_, y) + 0.1 * proximal_term
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.schedule.step()
        model = model.to("cpu")