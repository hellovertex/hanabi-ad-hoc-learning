import torch
import numpy as np
from cl2 import StatesCollector, AGENT_CLASSES
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

    def yield_batch(self):
        for i in range(self._max_iter):
            states, actions = self._data_collector.collect(self._batch_size, agent_id=self._agent_id)
            yield torch.from_numpy(states).type(torch.FloatTensor), torch.max(torch.from_numpy(actions), 1)[1]



    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 # max_size_per_iter: int = 100,
                 agent_id: Optional[str] = None,
                 batch_size: int = 1,
                 max_iter=1000  # number of batches returned by self.__iter__ until StopIteration
                 ):
        super(IterableStatesCollection).__init__()
        self._data_collector = StatesCollector(agent_classes, num_players, agent_id)
        # self._max_size_per_iter = max_size_per_iter
        self._agent_id = agent_id  # if None,
        # self.collect_data will return state,action pairs for all agents in AGENT_CLASSES
        self._batch_size = batch_size
        # assert max_size_per_iter > batch_size, "maximum number of states to collect must be larger than batch_size"
        self._max_iter = max_iter
    def __iter__(self):
        # collect some states in a list (streaming data)
        # doing this using __len__ and __getitem__ would also work:
        # __getitem__ should implement a check if num_states have been collected
        # and load new states otherwise but its not as nice as an iterable
        # return iter(self.collect_data(max_states=self._max_size_per_iter))
        # todo would work if collect yielded and not returned,
        # todo we can simulate this instead inside self.colled_data()
        # states, actions = self._data_collector.collect(self._max_size_per_iter)
        # return zip(states, actions)   # returns eagerly
        batch_size = 2
        # states, actions = [(x0,y0), (x1,y1), ..., (xn, yn)]
        # return zip(*[self.collect_data(self._max_size_per_iter), self.collect_data(self._max_size_per_iter)])
        # return iter([self.collect_data(self._max_size_per_iter) for _ in range(self._batch_size)])
        return iter(self.yield_batch())

#st = time()
#tr = ts = IterableStatesCollection(AGENT_CLASSES, num_players=3, agent_id="InternalAgent", batch_size=64)
#print(hasattr(tr, '__next__'), hasattr(tr, '__iter__'))
#for d in tr:
#    x, y = d
#    print(x.shape)
#    print(y.shape)
#    break

# todo filter state action pairs by agent
# loader: np.array([100, 956]), np.array([100, 30])
# want: np.array([100, 956]), np.array([100, 30]) = next(iter(loader))
