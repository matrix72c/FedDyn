import abc

class Client(metaclass=abc.ABCMeta):
    def __init__(self, train_data, train_label, user_num, item_num, run, client_id=0):
        self.train_data = train_data
        self.train_label = train_label
        self.user_num = user_num
        self.item_num = item_num
        self.run = run
        self.client_id = client_id
        self.lr_count = 0
    @abc.abstractmethod
    def train(self):
        pass