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
import model
import torch.optim as optim
AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}


class PoolOfStates(torch.utils.data.IterableDataset):
  def __init__(self, from_db_path='./database.db',
               load_state_as_type='dict',  # or 'dict'
               drop_actions=False,
               size=1e5,
               target_table='pool_of_states',
               batch_size=1,
               load_lazily=True):
    super(PoolOfStates).__init__()
    self._from_db_path = from_db_path  # path to .db file
    self._drop_actions = drop_actions
    self._size = size
    self._target_table = target_table
    self._connection = sqlite3.connect(self._from_db_path)
    self._batch_size = batch_size
    if batch_size != 1: raise NotImplementedError
    self._load_lazily = load_lazily
    assert load_state_as_type in ['torch.FloatTensor', 'dict'], 'states must be either torch.FloatTensor or dict'
    self._load_state_as_type = load_state_as_type
    self.QUERY_VARS = ['num_players',
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
    actions = [] if drop_actions else ['int_action', 'dict_action ']
    self.QUERY_VARS += actions

  class QueryCols(enum.IntEnum):
    num_players = 0
    agent = 1
    current_player = 2
    current_player_offset = 3
    deck_size = 4
    discard_pile = 5
    fireworks = 6
    information_tokens = 7
    legal_moves = 8
    life_tokens = 9
    observed_hands = 10
    card_knowledge = 11
    vectorized = 12
    pyhanabi = 13
    int_action = 14
    dict_action = 15

  def _build_query(self, table='pool_of_state_dicts') -> str:
    query_cols = [col + ', ' for col in self.QUERY_VARS]
    query_cols[-1] = query_cols[-1][:-2]  # remove last ,
    query_string = ['SELECT '] + query_cols + ['from ' + table]
    return "".join(query_string)

  def _parse_row_to_dict(self, row):
    obs_dict = {}
    # assign columns of query to corresponding key in observation_dict
    obs_dict[self.QueryCols.num_players.name] = row[self.QueryCols.num_players.value]
    obs_dict[self.QueryCols.agent.name] = row[self.QueryCols.agent.value]
    obs_dict[self.QueryCols.current_player.name] = row[self.QueryCols.current_player.value]
    obs_dict[self.QueryCols.current_player_offset.name] = row[self.QueryCols.current_player_offset.value]
    obs_dict[self.QueryCols.deck_size.name] = row[self.QueryCols.deck_size.value]
    obs_dict[self.QueryCols.discard_pile.name] = eval(row[self.QueryCols.discard_pile.value])
    obs_dict[self.QueryCols.fireworks.name] = eval(row[self.QueryCols.fireworks.value])
    obs_dict[self.QueryCols.information_tokens.name] = row[self.QueryCols.information_tokens.value]
    obs_dict[self.QueryCols.legal_moves.name] = eval(row[self.QueryCols.legal_moves.value])
    obs_dict[self.QueryCols.life_tokens.name] = row[self.QueryCols.life_tokens.value]
    obs_dict[self.QueryCols.observed_hands.name] = eval(row[self.QueryCols.observed_hands.value])
    obs_dict[self.QueryCols.card_knowledge.name] = eval(row[self.QueryCols.card_knowledge.value])
    obs_dict[self.QueryCols.vectorized.name] = eval(row[self.QueryCols.vectorized.value])
    obs_dict[self.QueryCols.pyhanabi.name] = pickle.loads(row[self.QueryCols.pyhanabi.value])
    if not self._drop_actions:
      obs_dict[self.QueryCols.int_action.name] = row[self.QueryCols.int_action.value]
      obs_dict[self.QueryCols.dict_action.name] = row[self.QueryCols.dict_action.value]

    return obs_dict

  def _yield_dict(self):
    cursor = self._connection.cursor()
    query_string = self._build_query()
    # query database with all the information necessary to build the observation_dictionary
    cursor.execute(query_string)
    # parse query
    for row in cursor:  # database row
      # build observation_dict from row
      obs_dict = self._parse_row_to_dict(row)
      # yield row by row the observation_dictionary unpacked from that row
      yield obs_dict

  def get_rows_lazily(self):
    if self._load_state_as_type == 'dict':
      return self._yield_dict()
    elif self._load_state_as_type == 'torch.FloatTensor':
      raise NotImplementedError
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
        print(f'Dict action in database = {state["dict_action"]}')
        print(state['agent'])

        break
    except Exception as e:
      print(type(e), e, i)
      traceback.print_exc()
      break

    # predicted = model(state)
    # update_model(action, predicted)


def train_eval(config,
               target_agent_cls,
               from_db_path='./database_test.db',
               target_table='pool_of_states'):
  # todo use ray to centralize dataloading
  dataset = PoolOfStates(from_db_path=from_db_path, batch_size=1, drop_actions=False, load_state_as_type='dict')
  dataloader = DataLoader(dataset, batch_size=None)

  target_agent = target_agent_cls(config)
  net = model.get_model(observation_size=956, num_actions=30, num_hidden_layers=1, layer_size=512)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)
  for state in dataloader:
    observation = state[0]
    action = target_agent.act(observation)
    # action = parse to torch.Tensor
    vectorized = observation['vectorized']
    optimizer.zero_grad()
    outputs = net(vectorized)
    loss = criterion(outputs, action)
    loss.backward()
    optimizer.step()

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
