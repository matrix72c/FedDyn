import time
from tqdm import tqdm
import yaml
from aim import Run
from clients import *
from servers import *
from utils import *

f = open('config.yaml', 'r', encoding='utf-8')
conf = yaml.load(f.read(), Loader=yaml.FullLoader)

run = Run(experiment='performance')
run['hparams'] = conf

client_name = run['hparams']['method'] + "Client"
server_name = run['hparams']['method'] + "Server"
Client_ = getattr(__import__("clients."+client_name, fromlist=[client_name]), run['hparams']['method']+"Client")
Server_ = getattr(__import__("servers."+server_name, fromlist=[server_name]), run['hparams']['method']+"Server")

user_num, item_num, train_data, train_label, test_data = get_ncf_data("./data/" + run['hparams']['dataset'] + ".train.rating", "./data/" + run['hparams']['dataset'] + ".test.negative", run['hparams']['neg_pos_ratio'])

clients_train_data, clients_train_label = [[] for _ in range(user_num)], [[] for _ in range(user_num)]
for i in range(len(train_data)):
    user, item = train_data[i][0], train_data[i][1]
    clients_train_data[user].append([user, item])
    clients_train_label[user].append(train_label[i])
client_list = []
for i in range(len(clients_train_data)):
    c = Client_(clients_train_data[i], clients_train_label[i], user_num, item_num, run, i)
    client_list.append(c)

t = time.time()
server = Server_(client_list, user_num, item_num, test_data, run)
for rnd in range(run['hparams']['rounds']):
    server.iterate()
    if rnd % run['hparams']['eval_every'] == 0:
        hit, ndcg = server.model.test(test_data, run['hparams']['top_k'])
        run.track(hit, name="hit", epoch=rnd)
        run.track(ndcg, name="ndcg", epoch=rnd)
        tqdm.write("Round: %d, Time: %.1fs, Hit: %.4f, NDCG: %.4f" % (rnd, time.time() - t, hit, ndcg))
        t = time.time()