import torch
import sqlite3
import numpy as np
from torch.utils.data import DataLoader
import pickle
# project lvl imports
import rulebased_agent as ra
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent

AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}


class PoolOfStates(torch.utils.data.IterableDataset):
    def __init__(self, from_db_path='./database.db',
                 load_state_as_type='torch.FloatTensor',  # or 'dict'
                 drop_actions=False,
                 size=1e5,
                 target_table='pool_of_states',
                 batch_size=4,
                 load_lazily=True):
        super(PoolOfStates).__init__()
        self._from_db_path = from_db_path  # path to .db file
        self._drop_actions = drop_actions
        self._size = size
        self._target_table = target_table
        self._connection = sqlite3.connect(self._from_db_path)
        self._batch_size = batch_size
        self._load_lazily = load_lazily
        assert load_state_as_type in ['torch.FloatTensor', 'dict'], 'states must be either torch.FloatTensor or dict'
        self._load_state_as_type = load_state_as_type

    @staticmethod
    def _tolist(state: str, is_tuple=False):
        if is_tuple:
            state = state[0]
        state = eval(state)
        assert isinstance(state, list)
        return state

    def _parse_state(self, state: str, is_tuple=False):
        state = self._tolist(state, is_tuple)
        return torch.from_numpy(np.array(state))
    #
    # def _parse_dict(self, db_row, action=False):
    #     if action:
    #         current_player, \
    #         current_player_offset, \
    #         deck_size, \
    #         discard_pile, \
    #         fireworks, \
    #         information_tokens, \
    #         legal_moves, \
    #         life_tokens, \
    #         observed_hands, \
    #         card_knowledge, \
    #         pyhanabi, \
    #         num_players, \
    #         state, \
    #         action = db_row
    #     else:
    #         current_player, \
    #         current_player_offset, \
    #         deck_size, \
    #         discard_pile, \
    #         fireworks, \
    #         information_tokens, \
    #         legal_moves, \
    #         life_tokens, \
    #         observed_hands, \
    #         card_knowledge, \
    #         pyhanabi, \
    #         num_players, \
    #         state = db_row

    def _yield_vectorized(self):
        cursor = self._connection.cursor()
        action = "" if self._drop_actions else ", action"

        cursor.execute(f'SELECT state{action} from pool_of_states')
        if self._drop_actions:
            for state in cursor:
                yield self._parse_state(state, is_tuple=True)
        else:
            for row in cursor:
                state, action = row
                yield self._parse_state(state), action

    def _yield_dict(self):
        cursor = self._connection.cursor()
        action = "" if self._drop_actions else ", action"
        obs_dict = {}
        cursor.execute(f'SELECT current_player, '
                       f'current_player_offset, '
                       f'deck_size, '
                       f'discard_pile, '
                       f'fireworks, '
                       f'information_tokens, '
                       f'legal_moves, '
                       f'life_tokens, '
                       f'observed_hands, '
                       f'card_knowledge, '
                       f'pyhanabi, '
                       f'num_players, '
                       f'state{action} from pool_of_state_dicts')
        if self._drop_actions:
            for row in cursor:
                current_player, current_player_offset, deck_size, discard_pile, fireworks, information_tokens, legal_moves, life_tokens, observed_hands, card_knowledge, pyhanabi, num_players, vectorized = row
                obs_dict['current_player'] = current_player
                obs_dict['current_player_offset'] = current_player_offset
                obs_dict['deck_size'] = deck_size
                obs_dict['discard_pile'] = eval(discard_pile)
                obs_dict['fireworks'] = eval(fireworks)
                obs_dict['information_tokens'] = information_tokens
                obs_dict['legal_moves'] = eval(legal_moves)
                obs_dict['life_tokens'] = life_tokens
                obs_dict['observed_hands'] = eval(observed_hands)
                obs_dict['card_knowledge'] = eval(card_knowledge)
                # obs_dict['pyhanabi'] = pickle.loads(pyhanabi)
                obs_dict['num_players'] = num_players
                obs_dict['vectorized'] = eval(vectorized)
                yield obs_dict
            else:
                raise NotImplementedError

    def get_rows_lazily(self):
        if self._load_state_as_type == 'torch.FloatTensor':
            return self._yield_vectorized()
        elif self._load_state_as_type == 'dict':
            return self._yield_dict()
        else:
            raise NotImplementedError

    @staticmethod
    def _create_batch(from_list):
        return zip(*from_list)

    def __iter__(self):
        if self._load_lazily:
            return iter(self._create_batch([self.get_rows_lazily() for _ in range(self._batch_size)]))
        else:
            # todo here, we could load greedily to distribute a large dataset via ray
            # maybe this will speed up the process by removing the dataloading bottleneck on the workers
            raise NotImplementedError


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


def train_eval(config,
               target_agent_cls,
               from_db_path='./database_test.db',
               target_table='pool_of_states'):
    # pool of states
    # todo consider using ray to centralize dataloading to avoid racing for database
    dataset = PoolOfStates(from_db_path=from_db_path, batch_size=1, drop_actions=True, load_state_as_type='dict')
    dataloader = DataLoader(dataset, batch_size=None)

    target_agent = target_agent_cls(config)
    # model = Model(config)
    i = 0
    for state in dataloader:
        if i == 1:
            break
        state = state[0]
        print(state)
        # action = target_agent.act(state)
        # print(f'action was {action}')
        i += 1

        # predicted = model(state)
        # update_model(action, predicted)

def main():
    # todo include num_players to sql query
    num_players = 3
    config = {'players': num_players}
    target_agent_cls = InternalAgent
    train_eval(config=config,
               target_agent_cls=target_agent_cls)


def test_pickling():
    conn = sqlite3.connect('./database_test.db')
    cursor = conn.cursor()
    cursor.execute('SELECT pyhanabi from pool_of_state_dicts')
    for row in cursor:
        print(row)
        pyhanabi = pickle.loads(row[0])
        print(pyhanabi)
        break

if __name__ == '__main__':
    # main()
    test_pickling()
