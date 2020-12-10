import torch
import numpy as np
from cl2 import StateActionCollector, AGENT_CLASSES
from typing import Dict, Optional
from time import time
from itertools import cycle, chain
import random
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


class IterableStatesCollection(torch.utils.data.IterableDataset):
    # the iter() call of the dataset streams data to the loader who can batch, shuffle, transform it
    # to create a new iterator, and by calling iter() on the loader it returns an instance of
    # torch.utils.data._BaseDataLoaderIter which happens to implement __next__ and is thus a generator

    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 # max_size_per_iter: int = 100,
                 target_agent: Optional[str] = None,
                 batch_size: int = 1,
                 len_iter=1000  # number of batches returned by self.__iter__ until StopIteration
                 ):
        super(IterableStatesCollection).__init__()
        self._data_collector = StateActionCollector(agent_classes, num_players, target_agent)
        self._target_agent = target_agent  # if None, target agent wont necessarily be a player in collect-games
        self._batch_size = batch_size
        self._len_iter = len_iter

    def yield_batch(self):
        # todo sample datasets not batches to pass  to __iter__, its much faster
        # todo because now we reset environment on each batch
        for i in range(self._len_iter):
            states, actions = self._data_collector.collect(self._batch_size, agent_id=self._target_agent)
            yield torch.from_numpy(states).type(torch.FloatTensor), torch.max(torch.from_numpy(actions), 1)[1]

    def get_states(self):
        states, actions = self._data_collector.collect(drop_actions=False,
                                                       max_states=self._len_iter,
                                                       agent_id=self._target_agent,
                                                       games_per_group=10)
        # states, actions = torch.from_numpy(states).type(torch.FloatTensor), torch.max(torch.from_numpy(actions), 1)[1]
        states, actions = torch.from_numpy(states).type(torch.FloatTensor), torch.from_numpy(actions).type(
            torch.LongTensor)
        n = int(self._len_iter / self._batch_size)
        for s, a in zip([states[i:i + self._batch_size] for i in range(n)],
                        [actions[i:i + self._batch_size] for i in range(n)]):
            yield s, a

    def __iter__(self):
        return iter(self.get_states())


class StateActionWriter:
    """ - Collects states and actions using the StateActionCollector
        - writes them to database
    """

    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 # max_size_per_iter: int = 100,
                 target_agent: Optional[str] = None):
        self._data_collector = StateActionCollector(agent_classes, num_players, target_agent)

    def collect_and_write_to_database(self, path_to_db, num_rows_to_add, use_state_dict=False):
        # if path_to_db does not exists, create a file, otherwise append to database
        #        x          x       x      x       x        x
        # | num_players | agent | turn | state | action | team |
        """ If use_state_dict is True, a different table will be used, that stores the
        observation as a dictionary """
        collected = 0
        while collected < num_rows_to_add:
            self._data_collector.collect(max_states=1000,  # only thousand at once because its slow otherwise
                                         insert_to_database_at=path_to_db,
                                         keep_obs_dict=use_state_dict)
            collected += 1000


def write(path_to_db, num_rows_to_add, use_state_dict=False):
    writer = StateActionWriter(AGENT_CLASSES, 3)
    writer.collect_and_write_to_database(path_to_db, num_rows_to_add, use_state_dict=use_state_dict)


def collect(num_states_to_collect):
    collector = StateActionCollector(AGENT_CLASSES, 3)
    states = collector.collect(drop_actions=True,
                               max_states=num_states_to_collect,
                               target_agent=None,
                               keep_obs_dict=False)
    return states
# write(path_to_db='./database_test.db', num_rows_to_add=500, use_state_dict=True)
print(collect(100))
