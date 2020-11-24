import torch
import numpy as np
from cl2 import StateActionCollector, AGENT_CLASSES
from typing import Dict, Optional
import rulebased_agent as ra
from time import time
from itertools import cycle, chain
from internal_agent import InternalAgent
import random


class IterableStatesCollection(torch.utils.data.IterableDataset):
    # the iter() call of the dataset streams data to the loader who can batch, shuffle, transform it
    # to create a new iterator, and by calling iter() on the loader it returns an instance of
    # torch.utils.data._BaseDataLoaderIter which happens to implement __next__ and is thus a generator

    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 # max_size_per_iter: int = 100,
                 agent_id: Optional[str] = None,
                 batch_size: int = 1,
                 len_iter=1000  # number of batches returned by self.__iter__ until StopIteration
                 ):
        super(IterableStatesCollection).__init__()
        self._data_collector = StateActionCollector(agent_classes, num_players, agent_id)
        self._agent_id = agent_id  # if None, target agent wont necessarily be a player in collect-games
        self._batch_size = batch_size
        self._len_iter = len_iter

    def yield_batch(self):
        # todo sample datasets not batches to pass  to __iter__, its much faster
        # todo because now we reset environment on each batch
        for i in range(self._len_iter):
            states, actions = self._data_collector.collect(self._batch_size, agent_id=self._agent_id)
            yield torch.from_numpy(states).type(torch.FloatTensor), torch.max(torch.from_numpy(actions), 1)[1]

    def get_states(self):
        states, actions = self._data_collector.collect(drop_actions=False,
                                                       max_states=self._len_iter,
                                                       agent_id=self._agent_id,
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
# todo filter state action pairs by agent
