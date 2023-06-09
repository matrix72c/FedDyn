import copy
import random
import time
from interface import server
from model import NCFModel

class FedAvgServer(server.Server):
    def __init__(self, clients, user_num, item_num, test_data, run):
        self.model = NCFModel(user_num, item_num, run['hparams']['factor_num'], run['hparams']['num_layers'], run['hparams']['dropout'], run['hparams']['model_type'])
        super(FedAvgServer, self).__init__(clients, user_num, item_num, test_data, run)
    
    def iterate(self):
        t = time.time()
        clients = random.sample(self.clients, self.run['hparams']['sample_size'])
        for client in clients:
            client.model.load_state_dict(self.model.state_dict())
            client.train()
        server_dict = copy.deepcopy(clients[0].model.state_dict())
        for i in range(1, len(clients)):
            client_dict = clients[i].model.state_dict()
            for k in client_dict.keys():
                server_dict[k] += client_dict[k]
        for k in server_dict.keys():
            server_dict[k] /= len(clients)
        self.model.load_state_dict(server_dict)