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
import traceback
import enum

AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}


class QueryColumns(enum.IntEnum):
  """ Columns sorted according to database layout """
  # todo remove this
  num_players = 0
  agent = 1
  turn = 2
  int_action = 3
  dict_action = 4
  team = 5
  current_player = 6
  current_player_offset = 7
  deck_size = 8
  discard_pile = 9
  fireworks = 10
  information_tokens = 11
  legal_moves = 12
  life_tokens = 13
  observed_hands = 14
  card_knowledge = 15
  vectorized = 16
  pyhanabi = 17


class PoolOfStates(torch.utils.data.IterableDataset):
  def __init__(self, from_db_path='./database.db',
               load_state_as_type='dict',  # or 'dict'
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
    self.query_vars = ['num_players',
                        'agent',
                        'current_player',
                        'current_player_offset',
                        'deck_size',
                        'discard_pile',
                        'fireworks',
                        'information_tokens',
                        'legal_moves',
                        'life_tokens',
                        'observed_hands',
                        'card_knowledge',
                        'vectorized',
                        'pyhanabi']
    actions = [] if self._drop_actions else ['int_action', 'dict_action ']
    self.query_vars += actions

  @staticmethod
  def _build_query(query_columns: list, table='pool_of_state_dicts') -> str:
    query_cols = [col+ ', ' for col in query_columns]
    query_cols[-1] = query_cols[-1][:-2]  # remove last ,
    query_string = ['SELECT '] + query_cols + ['from ' + table]
    return "".join(query_string)

  def _yield_dict(self):
    cursor = self._connection.cursor()
    obs_dict = {}
    query_string = self._build_query(self.query_vars)
    # query database with all the information necessary to build the observation_dictionary
    cursor.execute(query_string)
    # parse query
    for row in cursor:  # database row
      for i, var in enumerate(self.query_vars):  # assign columns of query to corresponding key in observation_dict
        obs_dict[var] = row[i]
      # yield row by row the observation_dictionary unpacked from that row
      yield obs_dict

  def get_rows_lazily(self):
    if self._load_state_as_type == 'torch.FloatTensor':
      raise NotImplementedError
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


class Model:
  def __init__(self, config):
    self.config = config

  def __call__(self, *args, **kwargs):
    return 1


def update_model(*args):
  pass


def int_action_to_dict(int_action):
  card_index = -1
  target_offset = -1
  action_type = None
  color = -1
  rank = -1
  if 0 <= int_action <= 4:
    action_type = 'PLAY'
  elif 5 <= int_action <= 9:
    action_type = 'DISCARD'
  elif 10 <= int_action <= 29:
    action_type = 'REVEAL_COLOR'
  elif 30 <= int_action <= 49:
    action_type = 'REVEAL_RANK'
  if action_type == 'REVEAL_COLOR':
    color = int_action % 10
    target_offset = int(int_action / 10)
  elif action_type == 'REVEAL_RANK':
    rank = int_action % 10
    target_offset = int(int_action / 10) - 2
  return {'action_type': action_type, 'card_index': card_index, 'target_offset': target_offset, 'color': color,
          'rank': rank}


def train_eval_test(config,
                    target_agent_cls,
                    from_db_path='./database_test.db',
                    target_table='pool_of_states'):
  # pool of states
  # todo consider using ray to centralize dataloading to avoid racing for database
  dataset = PoolOfStates(from_db_path=from_db_path, batch_size=1, drop_actions=False, load_state_as_type='dict')
  dataloader = DataLoader(dataset, batch_size=None)

  target_agent = target_agent_cls(config)
  # model = Model(config)
  i = 0
  for state in dataloader:
    try:
      state = state[0]
      # print(state)
      print(i)
      i += 1
      action = target_agent.act(state)
      if state['agent'] in str(target_agent_cls):
        print(f'action agent took online was {action}')
        print(f'int action in database = {state["int_action"]}')
        print(f'Dict action in database = {state["dict_action"]}')
        # print(f'converted int action =  {int_action_to_dict(state["int_action"])}')
        print(state['agent'])

        break
    except Exception as e:
      print(type(e), e, i)
      traceback.print_exc()
      break

    # predicted = model(state)
    # update_model(action, predicted)


def main():
  # todo include num_players to sql query
  num_players = 3
  config = {'players': num_players}
  target_agent_cls = VanDenBerghAgent
  train_eval_test(config=config, target_agent_cls=target_agent_cls)


def test_pickling():
  conn = sqlite3.connect('./database_test.db')
  cursor = conn.cursor()
  cursor.execute('SELECT pyhanabi from pool_of_state_dicts')
  i = 0
  for row in cursor:
    # print(row)
    try:
      pyhanabi = pickle.loads(row[0])
      print(pyhanabi)
      print('pyhanabi')
      i += 1
    except Exception as e:
      print(i)
      print(e)
      break
    if i == 1000:
      print(i)
      break


DEBUG = False
if __name__ == '__main__':
  if DEBUG:
    test_pickling()
  else:
    main()
