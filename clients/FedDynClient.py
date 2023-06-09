from interface import client
import torch
import torch.utils.data as Data
from model import NCFModel
from utils import *
from torch.optim.lr_scheduler import StepLR
from torch import nn

class FedDynClient(client.Client):
    def __init__(self, train_data, train_label, user_num, item_num, run, client_id=0):
        self.model = NCFModel(user_num, item_num, run['hparams']['factor_num'], run['hparams']['num_layers'], run['hparams']['dropout'], run['hparams']['model_type'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=run['hparams']['learning_rate'])
        self.schedule = StepLR(self.optimizer, step_size=run['hparams']['lr_step'], gamma=run['hparams']['lr_decay'])
        self.dataset = NCFDataset(torch.tensor(train_data).to(torch.long), torch.tensor
        (train_label).to(torch.float32))
        self.loader = Data.DataLoader(self.dataset, batch_size=run['hparams']['batch_size'], shuffle=True, num_workers=0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = run['hparams']['epochs']
        self.lr = run['hparams']['learning_rate']
        self.lr_step = self.epochs
        self.lr_decay = run['hparams']['lr_decay']
        super(FedDynClient, self).__init__(train_data, train_label, user_num, item_num, run, client_id)
    
    def train(self):
        model = self.model.to(self.device)
        model.train()
        for epoch in range(self.epochs):
            batch_loss_list = []
            for data in self.loader:
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                y_ = self.model(x)
                loss = nn.BCEWithLogitsLoss()(y_, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss_list.append(loss.detach().item())
            self.schedule.step()
            mean_loss = np.mean(batch_loss_list)
            self.lr_count += 1
            if self.lr_count == self.lr_step:
                self.lr_count = 0
                self.lr = self.lr * self.lr_decay
                self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            # early stop
            if mean_loss < self.run['hparams']['local_loss_threshold']:
                break
        model = model.to("cpu")

    def get_distill_batch(self):
        model = self.model.to(self.device)
        model.eval()
        total_data = torch.tensor([[self.client_id, i] for i in range(self.item_num)])
        total_logits = []
        total_dataset = NCFDataset(total_data, [1. for _ in range(self.item_num)])
        total_dataloader = Data.DataLoader(total_dataset, batch_size=self.run['hparams']['batch_size'], shuffle=False)
        for data, label in total_dataloader:
            data = data.to(self.device)
            pred = model(data)
            total_logits.extend(pred.detach().cpu().numpy())
        total_logits = torch.tensor(total_logits)
        model.to("cpu")

        # get positive items
        num_positive = int(self.run['hparams']['distill_batch_size'] * self.run['hparams']['distill_pos_ratio'])
        _, indices = torch.topk(total_logits, num_positive)
        positive_data = total_data[indices]
        positive_logits = total_logits[indices]

        # get the rest of items
        total_data = torch_delete(total_data, indices)
        total_logits = torch_delete(total_logits, indices)

        # get neg items id and corresponding logits
        neg_samples = torch.randint(0, len(total_data), (self.run['hparams']['distill_batch_size'] - num_positive,))
        negative_data = total_data[neg_samples]
        negative_logits = total_logits[neg_samples]

        # concat positive and negative samples
        client_batch = torch.cat([positive_data, negative_data], dim=0)
        client_logits = torch.cat([positive_logits, negative_logits], dim=0)
        return client_batch, client_logits