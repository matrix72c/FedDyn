import abc

class Server(metaclass=abc.ABCMeta):
    def __init__(self, clients, user_num, item_num, test_data, run):
        self.clients = clients
        self.user_num = user_num
        self.item_num = item_num
        self.test_data = test_data
        self.run = run

    @abc.abstractmethod
    def iterate(self):
        pass