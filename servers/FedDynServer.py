import random
import time
import numpy as np

import torch
from interface import server
from model import NCFModel
import torch.utils.data as Data
from torch import nn
from torch.optim.lr_scheduler import StepLR

class FedDynServer(server.Server):
    def __init__(self, clients, user_num, item_num, test_data, run):
        self.model = NCFModel(user_num, item_num, run['hparams']['factor_num'], run['hparams']['num_layers'], run['hparams']['dropout'], run['hparams']['model_type'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distill_loss_func = nn.KLDivLoss(reduction='batchmean')
        self.distill_optimizer = torch.optim.Adam(self.model.parameters(), lr=run['hparams']['distill_learning_rate'])
        self.schedule = StepLR(self.distill_optimizer, step_size=run['hparams']['distill_lr_step'], gamma=run['hparams']['distill_lr_decay'])
        super(FedDynServer, self).__init__(clients, user_num, item_num, test_data, run)
    
    def iterate(self):
        distill_loss = None
        distill_batch = None
        distill_logits = None
        model = self.model.to(self.device)
        t = time.time()
        clients = random.sample(self.clients, self.run['hparams']['sample_size'])
        for client in clients:
            client.model.load_state_dict(self.model.state_dict())
            client.train()
        for client in clients:
            client_batch, client_logits = client.get_distill_batch()
            if distill_batch is None:
                distill_batch = client_batch
                distill_logits = client_logits
            else:
                distill_batch = torch.cat((distill_batch, client_batch), dim=0)
                distill_logits = torch.cat((distill_logits, client_logits), dim=0)
        distill_data = Data.TensorDataset(distill_batch, distill_logits)
        distill_loader = Data.DataLoader(distill_data, batch_size=self.run['hparams']['batch_size'], shuffle=True, drop_last=True)
        for _ in range(self.run['hparams']['distill_epochs']):
            distill_batch_loss_list = []
            for batch, logits in distill_loader:
                batch = batch.to(self.device)
                logits = logits.to(self.device)

                self.distill_optimizer.zero_grad()
                predict = model(batch)
                logits_softmax = torch.softmax(logits / self.run['hparams']['distill_T'], dim=0)
                predict_softmax = torch.softmax(predict / self.run['hparams']['distill_T'], dim=0)
                no_zero = torch.where(predict_softmax == 0, predict_softmax + 1e-10, predict_softmax)
                batch_loss = self.distill_loss_func(no_zero.log(), logits_softmax)
                batch_loss.backward()
                self.distill_optimizer.step()
                distill_batch_loss_list.append(batch_loss.item())
            distill_loss = np.mean(distill_batch_loss_list)
            if distill_loss < self.run['hparams']['distill_loss_threshold']:
                break
        self.schedule.step()
        model.to('cpu')