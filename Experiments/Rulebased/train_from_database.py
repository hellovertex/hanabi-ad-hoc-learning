from functools import partial

import torch
import sqlite3
import numpy as np
from torch.utils.data import DataLoader
import pickle
# project lvl imports
import rulebased_agent as ra
from Experiments.Rulebased.cl2 import StateActionCollector
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
from cl2 import AGENT_CLASSES
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune import CLIReporter
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}

DEBUG = False
USE_RAY = True
if DEBUG:
  LOG_INTERVAL = 10
  EVAL_INTERVAL = 20
  NUM_EVAL_STATES = 500
  # for model selection
  GRACE_PERIOD = 1
  MAX_T = 100
  MAX_TRAIN_ITERS = 1000
  NUM_SAMPLES = 1
  BATCH_SIZE = 4
else:
  LOG_INTERVAL = 100
  EVAL_INTERVAL = 500
  NUM_EVAL_STATES = 300
  GRACE_PERIOD = int(1e4)  # tune.report gets only called at EVAL_INTERVAL anyway
  MAX_T = int(110e3)  # tune.report gets only called at EVAL_INTERVAL anyway
  NUM_SAMPLES = 10
  MAX_TRAIN_ITERS = 100000
  BATCH_SIZE = 1  # tune.grid_search([8, 1])

KEEP_CHECKPOINTS_NUM = 50
VERBOSE = 1
from_db_path_notebook = '/home/cawa/Documents/github.com/hellovertex/hanabi-ad-hoc-learning/Experiments/Rulebased/database_test.db'
from_db_path_desktop = '/home/hellovertex/Documents/github.com/hellovertex/hanabi-ad-hoc-learning/Experiments/Rulebased/database_test.db'
FROM_DB_PATH = from_db_path_desktop

# todo load from config
hand_size = 5
num_players = 3
num_colors = 5
num_ranks = 5
COLORS = ['R', 'Y', 'G', 'W', 'B']
COLORS_INV = ['B', 'W', 'G', 'Y', 'R']
RANKS_INV = [4, 3, 2, 1, 0]
color_offset = (2 * hand_size)
rank_offset = color_offset + (num_players - 1) * num_colors


def to_int(action_dict):
  action_type = action_dict['action_type']
  if action_type == 'DISCARD':
    return action_dict['card_index']
  elif action_type == 'PLAY':
    return hand_size + action_dict['card_index']
  elif action_type == 'REVEAL_COLOR':
    return color_offset + action_dict['target_offset'] * num_colors - (COLORS_INV.index(action_dict['color'])) - 1
  elif action_type == 'REVEAL_RANK':
    return rank_offset + action_dict['target_offset'] * num_ranks - (RANKS_INV[action_dict['rank']]) - 1
  else:
    raise ValueError(f'action_dict was {action_dict}')


class PoolOfStatesFromDatabase(torch.utils.data.IterableDataset):
  def __init__(self, from_db_path='./database_test.db',
               load_state_as_type='dict',  # or 'dict'
               drop_actions=False,
               size=1e5,
               target_table='pool_of_state_dicts',
               batch_size=1,
               load_lazily=True):
    super(PoolOfStatesFromDatabase).__init__()
    self._from_db_path = from_db_path  # path to .db file
    self._drop_actions = drop_actions
    self._size = size
    self._target_table = target_table
    self._connection = sqlite3.connect(self._from_db_path)
    self._batch_size = batch_size
    # if batch_size != 1: raise NotImplementedError
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
    query_cols[-1] = query_cols[-1][:-2]  # remove last ', '
    query_string = ['SELECT '] + query_cols + [' from ' + table]
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
      # todo here, we could load eagerly to distribute a large dataset via ray.tune.run_with_parameters()
      # but I think the lazy implementation is parallelized quite nicely, because of the yield
      raise NotImplementedError


def eval_fn(net, eval_loader, criterion, target_agent, num_states):
  # load pickled observations and get vectorized and compute action and eval with that
  start = time()
  observations_pickled = eval_loader.collect(num_states_to_collect=num_states,
                                             keep_obs_dict=True,
                                             keep_agent=False)
  assert len(
    observations_pickled) == num_states, f'len(observations_pickled)={len(observations_pickled)} and num_states = {num_states}'
  # print(f'Collecting eval states took {time() - start} seconds')
  correct = 0
  running_loss = 0
  with torch.no_grad():
    for obs in observations_pickled:
      observation = pickle.loads(obs)
      action = torch.LongTensor([to_int(target_agent.act(observation))])
      prediction = net(torch.FloatTensor(observation['vectorized'])).reshape(1, -1)
      # loss
      running_loss += criterion(prediction, action)
      # accuracy
      correct += torch.sum(torch.max(prediction, 1)[1] == action)

    return 100 * running_loss / num_states, 100 * correct.item() / num_states


def train_eval(config,
               conn=None,
               checkpoint_dir=None,
               from_db_path=None,
               target_table='pool_of_state_dicts',
               log_interval=100,
               eval_interval=1000,
               num_eval_states=100,
               max_train_steps=np.inf,
               use_ray=True):
  target_agent_cls = config['agent']
  lr = config['lr']
  num_hidden_layers = config['num_hidden_layers']
  layer_size = config['layer_size']
  batch_size = config['batch_size']
  num_players = config['num_players']
  target_agent = target_agent_cls(config['agent_config'])

  if from_db_path is None:
    raise NotImplementedError("Todo: Implement the database setup before training on new machines. ")
  trainset = PoolOfStatesFromDatabase(from_db_path=from_db_path,
                                      target_table=target_table,
                                      batch_size=batch_size,
                                      drop_actions=True,
                                      load_state_as_type='dict')
  trainloader = DataLoader(trainset, batch_size=None)
  testloader = StateActionCollector(AGENT_CLASSES, 3)

  net = model.get_model(observation_size=956,  # todo derive this from game_config
                        num_actions=30,
                        num_hidden_layers=num_hidden_layers,
                        layer_size=layer_size)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=lr)
  it = 0
  if checkpoint_dir:
    print("Loading from checkpoints")
    path = os.path.join(checkpoint_dir, "checkpoint")
    model_state, optimizer_state = torch.load(path)
    net.load_state_dict(model_state())
    optimizer.load_state_dict(optimizer_state())
  epoch = 1
  moving_acc = 0
  eval_it = 0
  if not use_ray:
    ckptdir = checkpoint_dir if checkpoint_dir else ''
    writer = SummaryWriter(log_dir=ckptdir + 'manual_train/')

  while True:
    try:
      for batch in trainloader:
        # observation = batch[0]
        # action = target_agent.act(observation)
        # action = torch.LongTensor([to_int(action)])
        # actions = torch.LongTensor([to_int(target_agent.act(obs)) for obs in batch])
        # vectorized = torch.FloatTensor(observation['vectorized'])
        # vectorized = torch.FloatTensor([obs['vectorized'] for obs in batch])
        actions = torch.LongTensor([to_int(target_agent.act(obs)) for obs in batch])
        vectorized = torch.FloatTensor([obs['vectorized'] for obs in batch])
        optimizer.zero_grad()
        outputs = net(vectorized).reshape(batch_size, -1)
        loss = criterion(outputs, actions)

        loss.backward()
        optimizer.step()

        if it % log_interval == 0 and not use_ray:
          print(f'Iteration {it}...')
        if it % eval_interval == 0:
          loss, acc = eval_fn(net=net, eval_loader=testloader, criterion=criterion, target_agent=target_agent,
                              num_states=num_eval_states)
          moving_acc += acc
          eval_it += 1
          if not use_ray:
            # print(f'Loss at iteration {it} is {loss}, and accuracy is {moving_acc / eval_it} %')
            print(f'Loss at iteration {it} is {loss}, and accuracy is {acc} %')
            writer.add_scalar('Loss/test', loss, it)
            writer.add_scalar('Accuracy/test', acc, it)
            if checkpoint_dir:
              print(f'saving model to {checkpoint_dir}')
              path = os.path.join(checkpoint_dir, f'checkpoint_+{it}')
              torch.save((net.state_dict, optimizer.state_dict), path)
          else:
            # tune.report(training_iteration=it, loss=loss, acc=moving_acc / eval_it)
            tune.report(training_iteration=it, loss=loss, acc=acc)
            # checkpoint frequency may be handled by ray if we remove checkpointing here
            with tune.checkpoint_dir(step=it) as checkpoint_dir:
              path = os.path.join(checkpoint_dir, 'checkpoint')
              torch.save((net.state_dict, optimizer.state_dict), path)
        it += 1
        if it > max_train_steps:
          return
    except Exception as e:
      if isinstance(e, StopIteration):
        # database has been read fully, start over
        trainloader = DataLoader(trainset, batch_size=None)
        epoch += 1
        continue
      else:
        print(e)
        print(traceback.print_exc())
        raise e


def run_train_eval_with_ray(name, scheduler, search_space, metric, mode, log_interval, eval_interval, num_eval_states,
                            num_samples, max_train_steps):
  train_fn = partial(train_eval,
                     from_db_path=FROM_DB_PATH,
                     target_table='pool_of_state_dicts',
                     log_interval=log_interval,
                     eval_interval=eval_interval,
                     num_eval_states=num_eval_states,
                     max_train_steps=max_train_steps,
                     use_ray=True
                     )

  analysis = tune.run(train_fn,
                      metric=metric,
                      mode=mode,
                      config=search_space,
                      name=name,
                      num_samples=num_samples,
                      keep_checkpoints_num=KEEP_CHECKPOINTS_NUM,
                      verbose=VERBOSE,
                      trial_dirname_creator=None,
                      trial_name_creator=None,
                      # stopipp=ray.tune.EarlyStopping(metric='acc', top=5, patience=1, mode='max'),
                      scheduler=scheduler,
                      progress_reporter=CLIReporter(metric_columns=["loss", "acc", "training_iteration"]),
                      # trial_dirname_creator=trial_dirname_creator_fn
                      )
  best_trial = analysis.get_best_trial("acc", "max")
  print(best_trial.config)
  print(best_trial.checkpoint)

  # print(analysis.best_dataframe['acc'])
  return analysis


def select_best_model(name,
                      agentcls,
                      metric,
                      mode,
                      grace_period,
                      max_t,
                      num_samples,
                      lr,
                      layer_size,
                      batch_size):
  scheduler = ASHAScheduler(time_attr='training_iteration',
                            # metric=metric,
                            grace_period=grace_period,
                            # mode=mode,
                            max_t=max_t)  # current implementation raises stop iteration when db is finished
  # todo if necessary, build the search space from call params
  search_space = {'agent':  agentcls,  # tune.grid_search(list(AGENT_CLASSES.values())),  #  agentcls,
                  'lr': lr,  # learning rate seems to be best in [2e-3, 4e-3], old [1e-4, 1e-1]
                  'num_hidden_layers': 1,  # tune.grid_search([1, 2]),
                  # 'layer_size': tune.grid_search([64, 96, 128, 196, 256, 376, 448, 512]),
                  # 'layer_size': tune.grid_search([64, 96, 128, 196, 256]),
                  # 'layer_size': tune.grid_search([64, 128, 256]),
                  'layer_size': layer_size,
                  'batch_size': batch_size,  # tune.choice([4, 8, 16, 32]),
                  'num_players': num_players,
                  'agent_config': {'players': num_players}
                  }
  return run_train_eval_with_ray(name=name,
                                 scheduler=scheduler,
                                 search_space=search_space,
                                 metric=metric,  # handled by scheduler, mutually exclusive
                                 mode=mode,  # handled by scheduler, mutually exclusive
                                 log_interval=LOG_INTERVAL,
                                 eval_interval=EVAL_INTERVAL,
                                 num_eval_states=NUM_EVAL_STATES,
                                 num_samples=num_samples,
                                 max_train_steps=max(max_t + 1, MAX_TRAIN_ITERS)
                                 )


def tune_best_model(experiment_name, analysis, with_pbt, max_train_steps=1e6):
  # todo: careful there are still bugs here
  # load config from analysis_obj
  best_trial = analysis.get_best_trial("acc", "max")
  config = best_trial.config
  print(analysis.best_checkpoint)
  print(analysis.best_logdir)
  print(analysis.best_result)
  print(analysis.best_result_df)
  print(config)
  # create search space for pbt
  search_space = {
    "lr": tune.uniform(1e-2, 1e-4),
  }
  # create pbt scheduler, see https://docs.ray.io/en/master/tune/api_docs/schedulers.html
  pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    # metric="acc",
    # mode="max",
    perturbation_interval=500,  # every 10 `time_attr` units
    # (training_iterations in this case)
    hyperparam_mutations=search_space)
  # run pbt with final checkpoint dir
  return run_train_eval_with_ray(name=experiment_name,
                                 scheduler=pbt,
                                 search_space=config,
                                 metric='acc',  # handled by scheduler, mutually exclusive
                                 mode='max',  # handled by scheduler, mutually exclusive
                                 log_interval=LOG_INTERVAL,
                                 eval_interval=EVAL_INTERVAL,
                                 num_eval_states=NUM_EVAL_STATES,
                                 num_samples=NUM_SAMPLES,
                                 max_train_steps=max_train_steps
                                 )


def train_best_model(name, analysis, train_steps, from_db_path):
  # todo load train_eval fn with checkpoint_dir=best_trial.checkpoint_dir and continue training
  best_trial = analysis.get_best_trial('acc', 'max')
  checkpoint_dir = best_trial.checkpoint.value
  print(f'starting trainign in {checkpoint_dir}')
  config = best_trial.config
  # train_eval(config=config, checkpoint_dir=checkpoint_dir,use_ray=False, max_train_steps=train_steps, from_db_path=from_db_path, num_eval_states=NUM_EVAL_STATES)
  train_fn = partial(train_eval, max_train_steps=train_steps, from_db_path=from_db_path,
                     num_eval_states=NUM_EVAL_STATES)

  def _trial_dirname_creator(trial):
    return f'Manual_training_lr={trial.config["lr"]}' + f'_layer_size={trial.config["layer_size"]}'

  def _trial_name_creator(trial):
    return 'only_trial_' + str(trial.config['agent'])

  tune.run(train_fn,
           metric='acc',
           mode='max',
           config=config,
           name=name,
           num_samples=1,
           keep_checkpoints_num=1,
           verbose=VERBOSE,
           trial_dirname_creator=_trial_dirname_creator,
           trial_name_creator=None,
           # stopipp=ray.tune.EarlyStopping(metric='acc', top=5, patience=1, mode='max'),
           scheduler=None,
           progress_reporter=CLIReporter(metric_columns=["loss", "acc", "training_iteration"]),
           # trial_dirname_creator=trial_dirname_creator_fn
           )


def main():
  # train_fn = partial(train_eval,
  #                    # from_db_path='/home/cawa/Documents/github.com/hellovertex/hanabi-ad-hoc-learning/Experiments/Rulebased/database_test.db',
  #                    # target_table='pool_of_state_dicts',
  #                    # log_interval=10,
  #                    # eval_interval=20,
  #                    # num_eval_states=500,
  #                    # break_at_iteration=100,
  #                    # use_ray=True
  #                    )
  # print(tune.utils.diagnose_serialization(train_fn))
  # exit(0)
  for agentname, agentcls in AGENT_CLASSES.items():
    if USE_RAY:
      best_model_analysis = select_best_model(name=agentname,
                                              agentcls=agentcls,
                                              metric='acc',
                                              mode='max',
                                              grace_period=GRACE_PERIOD,
                                              max_t=MAX_T,
                                              num_samples=NUM_SAMPLES,
                                              lr=tune.loguniform(1e-2, 4e-3),
                                              layer_size=tune.grid_search([64, 128, 256]),
                                              batch_size=BATCH_SIZE)  # USE DEFAULTS for metric etc
      # todo maybe create two tune.Experiment instances for these
      print(f'Result written to {best_model_analysis.get_best_trial("acc", "max").checkpoint.value}')
      # train_best_model(name=agentname, analysis=best_model_analysis,
      #                  train_steps=5e5,
      #                  from_db_path=FROM_DB_PATH)
      # final_model_dir = tune_best_model(experiment_name=agentname, analysis=best_model_analysis, with_pbt=True).best_checkpoint
      print('exiting...')
      # print(f'Trained model weights and checkpoints are stored in {final_model_dir}')
    else:
      # todo include num_players to sql query
      num_players = 3
      # agentname = 'VanDenBerghAgent'
      config = {'agent': FlawedAgent,
                'lr': 2e-3,
                'num_hidden_layers': 1,
                'layer_size': 64,
                'batch_size': BATCH_SIZE,  # tune.choice([4, 8, 16, 32]),
                'num_players': num_players,
                'agent_config': {'players': num_players}
                }
      train_eval(config,
                 conn=None,
                 checkpoint_dir=None,
                 from_db_path=FROM_DB_PATH,
                 target_table='pool_of_state_dicts',
                 log_interval=LOG_INTERVAL,
                 eval_interval=EVAL_INTERVAL,
                 num_eval_states=NUM_EVAL_STATES,
                 max_train_steps=np.inf,
                 use_ray=False)


if __name__ == '__main__':
  main()
