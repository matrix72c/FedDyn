import numpy as np
import torch.nn as nn
import torch


class NCFModel(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model_type):
        super(NCFModel, self).__init__()
        self.dropout = dropout
        self.model_type = model_type

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model_type in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """

        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        user = x[:, 0]
        item = x[:, 1]
        if not self.model_type == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model_type == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model_type == 'GMF':
            concat = output_GMF
        elif self.model_type == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
    
    def test(self, test_data, top_k):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        hits = []
        ndcgs = []
        for case in test_data:
            x = torch.tensor(case[0]).to(torch.long).to(device)
            gt_item = case[1]
            prediction = self(x)
            _, indices = torch.topk(prediction, top_k)
            recommends = torch.take(x[:, 1], indices).cpu().numpy().tolist()
            hits.append(1 if gt_item in recommends else 0)
            ndcgs.append(np.reciprocal(np.log2(recommends.index(gt_item) + 2)) if gt_item in recommends else 0)
        self.to('cpu')
        return np.mean(hits), np.mean(ndcgs)