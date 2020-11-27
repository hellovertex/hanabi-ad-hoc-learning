import torch
import sqlite3
from torch.utils.data import DataLoader


class PoolOfStates(torch.utils.data.IterableDataset):
    def __init__(self, from_db_path='./database.db',
                 drop_actions=False,
                 size=1e5,
                 target_table='pool_of_states',
                 batch_size=4):
        self._from_db_path = from_db_path  # path to .db file
        self._drop_actions = drop_actions
        self._size = size
        self._target_table = target_table
        self._connection = sqlite3.connect(self._from_db_path)
        self._batch_size = batch_size

    def _tolist(self, state: str, is_tuple=False):
        if is_tuple:
            state = state[0]
        state = eval(state)
        assert isinstance(state, list)
        return state

    def get_rows_lazily(self):
        cursor = self._connection.cursor()
        action = "" if self._drop_actions else ", action"
        cursor.execute(f'SELECT state{action} from pool_of_states')
        if self._drop_actions:
            for state in cursor:
                yield self._tolist(state, is_tuple=True)
        else:
            for row in cursor:
                state, action = row
                yield self._tolist(state), action

    @staticmethod
    def _create_batch(from_list):
        return zip(*from_list)

    def __iter__(self):
        return iter(self._create_batch([self.get_rows_lazily() for _ in range(self._batch_size)]))


dataset = PoolOfStates(drop_actions=True)
dataloader = DataLoader(dataset, batch_size=None)


def dirtytest():
    for data in dataloader:
        state = data
        print(len(state))
        print(len(state[0]))
        break


class Model:
    def __init__(self, config):
        self.config = config

    def __call__(self, *args, **kwargs):
        return 1


def update_model(*args):
    pass


def train_eval(config, from_db_path, target_agent_cls):
    dataloader = PoolOfStates(from_db_path, drop_actions=True)
    target_agent = target_agent_cls()
    model = Model(config)
    for data in dataloader:
        state, _ = data  # or maybe loop: for state in dataloader
        action = target_agent(state)
        predicted = model(state)
        update_model(action, predicted)
