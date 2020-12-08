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
from collections import namedtuple
from typing import NamedTuple
import database as db
print(rl_env.__file__)
AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}

COLORS = ['R', 'Y', 'G', 'W', 'B']


def to_int(action_dict):
    # max_moves: 5 x play, 5 x discard, reveal color x 20=5c*4pl, reveal rank x 20=5r*4pl
    int_action = None

    if action_dict['action_type'] == 'PLAY':
        int_action = action_dict['card_index']  # 0-4 slots
    elif action_dict['action_type'] == 'DISCARD':
        int_action = 5 + action_dict['card_index']  # 5-9 slots
    elif action_dict['action_type'] == 'REVEAL_COLOR':
        int_action = 10 + COLORS.index(action_dict['color']) * action_dict['target_offset']  # 10-29 slots
    elif action_dict['action_type'] == 'REVEAL_RANK':
        int_action = 30 + action_dict['rank'] * action_dict['target_offset']  # 30-49 slots
    # todo: dont use 50 slots but determine them dynamically depending on numplayers and cards
    # one_hot_action = [0 for _ in range(self.environment.game.max_moves())]
    # one_hot_action = [0 for _ in range(50)]
    # one_hot_action[int_action] = 1
    # return one_hot_action
    # assert type(int_action) == int, f'action was {int_action} with acction_dict = {action_dict} and type = {type(int_action)}'
    # assert 0 <= int_action <= 50, f'action was {int_action}  with acction_dict = {action_dict}  and type = {type(int_action)}'
    return int(int_action)  # convert from 'numpy.int64' to python int


def to_one_hot(action_dict):
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


class Runner:
    # consider using factory method to access outer from inner, see e.g.
    # https://stackoverflow.com/questions/2024566/how-to-access-outer-class-from-an-inner-class
    def __init__(self, num_players):
        self.num_players = num_players
        self.environment = rl_env.make('Hanabi-Full', self.num_players)
        self.agent_config = {'players': self.num_players}  # same for all ra.RulebasedAgent instances
        self.zeros = [0 for _ in range(50)]  # used by one hot encoding action function

    def run(self,
            agents: List[NamedTuple],  # Agent('name', 'ra.RulebasedAgent()')
            max_games=1,
            target_agent: Optional[str] = None,
            drop_actions=False,
            keep_obs_dict=False):
        """
        agents: Agent instances used to play game
        max_games: number of games to collect at maximum
        agent_target: if provided, only state-action pairs for corresponding agent are returned
        drop_actions: if True, only states will be returned without corresponding actions
        keep_obs_dict: If true, returned replay_dict will also contain the observation_dict
        """

        i_game = 0
        # cum_states = []
        # cum_actions = []
        turns_played = 0
        team = []
        replay_dict = {}

        for agent in agents:
            team.append(agent.name)
            try:
                replay_dict[agent.name] = {'states': [], 'actions': [], 'turns': [], 'obs_dict': []}
            except:
                # goes here if we have more than one copy of the same agent(key)
                pass

        while i_game < max_games:
            # create game
            observations = self.environment.reset()
            done = False
            # players = agents  # players = [agent(self.agent_config) for agent in agents]
            turn_in_game_i = 0
            while not done:
                # play game
                for agent_index, agent in enumerate(agents):
                    # for agent_index, (agent_str, agent) in enumerate(players):
                    observation = observations['player_observations'][agent_index]
                    action = agent.instance.act(observation)
                    # determine current player action, other actions are None
                    if observation['current_player'] == agent_index:
                        assert action is not None
                        current_player_action = action
                        # maybe store vectorized state and one-hot encoded action
                        if target_agent is None or target_agent == agent.name:
                            # add binary encoding of player position, so that actions offsets
                            # can be understood by the network, use 3 bits to encode (remove this later)
                            binary_encoded_player_index = [int(i) for i in f'{agent_index:03b}']
                            # cum_states.append(observation['vectorized'] + binary_encoded_player_index)
                            # cum_actions.append(self.to_one_hot(current_player_action))
                            replay_dict[agent.name]['states'].append(
                                observation['vectorized'] + binary_encoded_player_index)
                            if not drop_actions:
                                replay_dict[agent.name]['actions'].append(to_int(current_player_action))
                            replay_dict[agent.name]['turns'].append(turn_in_game_i)
                            replay_dict[agent.name]['obs_dict'].append(observation)

                            turns_played += 1
                            turn_in_game_i += 1
                    else:
                        assert action is None
                # end of turn
                observations, reward, done, unused_info = self.environment.step(current_player_action)

            # end of game - increment game or state counter
            i_game += 1

        # cumulated states and actions across all games
        # return cum_states, cum_actions, turns_played
        return replay_dict, turns_played


# Agent = namedtuple('Agent', ['name', 'instance'])

class Agent(NamedTuple):
    name: str
    instance: ra.RulebasedAgent


class StateActionCollector:
    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 target_agent: Optional[str] = None
                 ):
        self.agent_classes = agent_classes  # pool of agents used to play
        self.num_players = num_players
        # self.states = []
        self._target_agent = target_agent
        self.runner = Runner(self.num_players)
        self.initialized_agents = {}  # Dict[str, namedtuple]
        self._replay_dict = {}

    def _initialize_all_agents(self):
        """
        set self.initialized_agents, so that run() calls wont re-initialize them every time
        and instead, their instances can be sampled for each game
        """
        initialized_agents = {}

        for agent_str, agent_cls in self.agent_classes.items():
            initialized_agents[agent_str] = Agent(name=agent_str,
                                                  instance=agent_cls({'players': self.num_players}))
        self.initialized_agents = initialized_agents  # Dict[str, NamedTuple]

    def get_players(self, k, target_agent: Optional[str] = None) -> List[NamedTuple]:
        """
        If target_agent is specified, it will be one of the players
        """
        players = []
        if target_agent:
            players.append(self.initialized_agents[target_agent])
        players += random.choices(list(self.initialized_agents.values()), k=k)
        return players

    statelist = List[List]
    actionlist = List

    def write_to_database(self, path, replay_dictionary, team, with_obs_dict):

        #        x          x       x      x       x        x
        # | num_players | agent | turn | state | action | team |
        # current_player: 0
        # current_player_offset: 0
        # deck_size: 40
        # discard_pile: []
        # fireworks: {}
        # information_tokens: 8
        # legal_moves: [{}, ..., {}]
        # life_tokens: 3
        # observed_hands: [[{},...,{}], ..., [{},...,{}]]
        # num_players: 2
        # vectorized:
        # pyhanabi

        # creates database at path, if it does not exist already
        conn = db.create_connection(path)
        # if table exists, appends dictionary data, otherwise it creates the table first and then inserts
        replay_dictionary['team'] = [agent.name for agent in team]
        replay_dictionary['num_players'] = self.num_players
        db.insert_data(conn, replay_dictionary, with_obs_dict)

    def collect(self,
                drop_actions=False,
                max_states=1000,
                target_agent=None,
                games_per_group=1,
                insert_to_database_at: Optional[str] = None,
                keep_obs_dict=False) -> Tuple[statelist, actionlist]:
        """
        Play Hanabi games to collect states (and maybe actions) until max_states are collected.

        If insert_to_database_at is provided, states and actions will be stored there (if valid path).
        Otherwise, numpy representations are returned as Tuple[List[List], List[List]], to be used
        as training data for NN training.

        To minimize bias,
        self.num_players are sampled uniformly from self.agent_classes
        for each game.

        This way, the set of states will be more heterogeneous, e.g. some come from
        later states in the game, some have different hinted cards, etc. depending on the
        agents used to play.

        if agent_id is provided, the states and actions returned correspond only to corresponding agent
        """
        if target_agent:
            assert target_agent in AGENT_CLASSES.keys(), "Unkown Agent identifier provided"
        num_states = 0
        cum_states = []
        cum_actions = []
        # initialize all agents only once, and then pass them over to run function
        # target_agent_index = self._initialize_all_agents(agent_id)
        self._initialize_all_agents()
        k = self.num_players - bool(target_agent)  # maybe save one spot for target agent

        while num_states < max_states:

            # play one game with randomly sampled agents
            players = self.get_players(k=k, target_agent=target_agent)
            replay_dictionary, num_turns_played = self.runner.run(players,
                                                                  max_games=games_per_group,
                                                                  target_agent=target_agent,
                                                                  drop_actions=drop_actions,
                                                                  keep_obs_dict=keep_obs_dict)
            # 1. write to database
            if insert_to_database_at:
                self.write_to_database(path=insert_to_database_at,
                                       replay_dictionary=replay_dictionary,
                                       team=players,
                                       with_obs_dict=keep_obs_dict)
            # Xor 2. keep data until return
            else:
                for key, val in replay_dictionary.items():
                    if not isinstance(cum_states, np.ndarray):
                        cum_states = np.array(val['states'])
                        cum_actions = np.array(val['actions'])
                    else:
                        try:
                            cum_states = np.concatenate((cum_states, val['states']))
                            cum_actions = np.concatenate((cum_actions, val['actions']))
                        except ValueError as e:
                            # goes here if states or actions are empty
                            # (because we dropped actions or the corresponding agent didnt make any moves)

                            # print(e)
                            # print(cum_states, cum_actions, agents, num_states)
                            # exit(1)
                            pass
            # cumulated stats
            num_states += num_turns_played

        if not insert_to_database_at:
            # return random subset of cum_states, cum_actions
            max_len = len(cum_states)
            indices = [random.randint(0, max_len - 1) for _ in range(max_states)]
            return cum_states[indices], cum_actions[indices]
