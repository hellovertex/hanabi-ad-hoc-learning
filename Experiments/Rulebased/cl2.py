from typing import List, Dict, Optional, Tuple
import numpy as np
import pickle
import random
import rl_env
import os
import torch
import torchvision
import rulebased_agent as ra
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent
import random

AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}

COLORS = ['R', 'Y', 'G', 'W', 'B']

class StatesCollector:
    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 agent_id: Optional[str] = None
                 ):
        self.agent_classes = agent_classes
        self.num_players = num_players
        self.states = []
        self._agent_id = agent_id

        class _Runner:
            # consider using factory method to access outer from inner, see e.g.
            # https://stackoverflow.com/questions/2024566/how-to-access-outer-class-from-an-inner-class
            def __init__(self, num_players):
                self.num_players = num_players
                self.environment = rl_env.make('Hanabi-Full', self.num_players)
                self.agent_config = {'players': self.num_players}  # same for all ra.RulebasedAgent instances

            def to_one_hot(self, action_dict):
                def _deprecated():
                    # this does not work as intended
                    move = self.environment._build_move(action_dict)
                    int_action = self.environment.game.get_move_uid(move)  # 0 <= move_uid < max_moves()
                    one_hot_action = [0 for _ in range(self.environment.game.max_moves())]
                    one_hot_action[int_action] = 1
                    return one_hot_action
                # max_moves: 5 x play, 5 x discard, reveal color x 20=5c*4pl, reveal rank x 20=5r*4pl
                int_action = None

                if action_dict['action_type'] == 'PLAY':
                    int_action = action_dict['card_index']  # 0-4 slots
                elif action_dict['action_type'] == 'DISCARD':
                    int_action = 5 + action_dict['card_index']  # 5-9 slots
                elif action_dict['action_type'] == 'REVEAL_COLOR':
                    int_action = 10 + COLORS.index(action_dict['color']) * action_dict['target_offset']  # 10-29 slots
                elif action_dict['action_type'] == 'REVEAL_RANK':
                    int_action = 30 + action_dict['rank'] * action_dict['target_offset']  # 10-29 slots
                # todo: dont use 50 slots but determine them dynamically depending on numplayers and cards
                # one_hot_action = [0 for _ in range(self.environment.game.max_moves())]
                one_hot_action = [0 for _ in range(50)]
                one_hot_action[int_action] = 1
                return one_hot_action

            def run(self, agents: List[ra.RulebasedAgent], max_games=1, agent_id=None):
                """
                agents: Agent Classes used to sample players from (uniformly at random for each game)
                max_games: number of games to collect at maximum
                agent_id: if provided, only state-action pairs for corresponding agent are returned
                """

                i_game = 0
                cum_states = []
                cum_actions = []
                turns_played = 0

                def _is_target_agent(agent_cls):
                    # todo check if state collection for specific agent works, and if so
                    # todo train neural nets using the collected data
                    if agent_id:
                        # determine if agent_id corresponds to target agent
                        keys = list(AGENT_CLASSES.keys())
                        vals = list(AGENT_CLASSES.values())
                        try:
                            return agent_id == keys[vals.index(agent_cls)]
                        except KeyError as e:
                            return False
                    return True

                while i_game < max_games:
                    # create game
                    observations = self.environment.reset()
                    done = False
                    players = [agent(self.agent_config) for agent in agents]

                    while not done:
                        # play game
                        for agent_index, agent in enumerate(players):
                            observation = observations['player_observations'][agent_index]
                            action = agent.act(observation)
                            # determine current player action, other actions are None
                            if observation['current_player'] == agent_index:
                                assert action is not None
                                current_player_action = action
                                # maybe store vectorized state and one-hot encoded action
                                if _is_target_agent(agent_cls=agents[agent_index]):
                                    # add binary encoding of player position, so that actions offsets
                                    # can be understood by the network, use 3 bits to encode
                                    binary_encoded_player_index = [int(i) for i in f'{agent_index:03b}']
                                    cum_states.append(observation['vectorized'] + binary_encoded_player_index)
                                    cum_actions.append(self.to_one_hot(current_player_action))
                                    turns_played += 1
                            else:
                                assert action is None
                        # end of turn
                        observations, reward, done, unused_info = self.environment.step(current_player_action)

                    # end of game - increment game or state counter
                    i_game += 1

                # cumulated states and actions across all games
                return cum_states, cum_actions, turns_played

        self.runner = _Runner(self.num_players)

    def collect(self, max_states=1000, sample_w_replacement=True, agent_id=None) -> Tuple[List[List], List[List]]:
        """
        Play Hanabi games and store their states until num_states have been stored.

        To minimize bias,
        self.num_players are sampled uniformly from self.agent_classes
        for each game.

        This way, the set of states will be more heterogeneous, e.g. some come from
        later states in the game, some have different hinted cards, etc. depending on the
        agents used to play.

        if agent_id is provided, the states and actions returned correspond only to corresponding agent
        """
        if agent_id:
            assert agent_id in AGENT_CLASSES.keys(), "Unkown Agent identifier provided"
        num_states = 0
        cum_states = []
        cum_actions = []

        def _sample_agents_uniformly_w_replacement():
            # init agent list
            if not agent_id:
                k = self.num_players
                agents = []
            else:
                k = self.num_players - 1
                agents = [self.agent_classes[agent_id]]

            # fill agent list
            if sample_w_replacement:
                agent_classes = random.choices(list(self.agent_classes.items()), k=k)
            else:
                agent_classes = random.sample(self.agent_classes.items(), k=k)

            for (unused_key, v) in agent_classes: agents.append(v)  # append class instance
            # random.shuffle(agents)
            return agents

        while num_states < max_states:

            # play one game with sampled agents
            agents = _sample_agents_uniformly_w_replacement()
            states, actions, num_turns_played = self.runner.run(agents, max_games=1, agent_id=agent_id)

            # cumulated stats
            num_states += num_turns_played
            if not isinstance(cum_states, np.ndarray):
                cum_states = np.array(states)
                cum_actions = np.array(actions)
            else:
                cum_states = np.concatenate((cum_states, states))
                cum_actions = np.concatenate((cum_actions, actions))

        # return random subset of cum_states, cum_actions
        max_len = len(cum_states)
        indices = [random.randint(0,max_len-1) for _ in range(max_states)]
        return cum_states[indices], cum_actions[indices]

    def save(self, to_path='./pickled_states', states=None):
        """ states default to self.states """
        with open(file=to_path + f'{self.num_players}_players', mode='wb') as f:
            if states is None:
                pickle.dump(self.states, f)
            else:
                assert type(states) == list
                pickle.dump(states, f)

    def load(self, from_path='./pickled_states'):
        if not os.path.exists(from_path):
            with open(file=from_path + f'{self.num_players}_players', mode='rb') as f:
                return pickle.load(f)
        else:
            with open(file=from_path, mode='rb') as f:
                return pickle.load(f)


# todo https://pytorch.org/docs/stable/data.html
# todo implement dataloader for states, actions and train neural net

class Generate:
    def __init__(self):
        self.last = 0

    def __iter__(self):
        self.last = 0
        return self

    def __next__(self):
        rv = self.last
        self.last += 1
        if self.last >= 10:
            raise StopIteration
        return rv


def generate():
    for i in range(10):
        yield i


f = generate()
g = Generate()
print(type(f), type(g))
#assert type(f) == type(g)
assert next(f) == next(g)


def load_data():
    """ Log the lengths of each training dataset here """
    LEN_MNIST = 10

    def _generate_1():
        for i in range(LEN_MNIST):
            yield i

    def _generate_2():
        pass
        # return iter(DataLoader)

    return _generate_1, _generate_1()


gen_train, gen_test = load_data()
print(hasattr(gen_train, "__iter__"))
print(hasattr(gen_train, "__next__"))