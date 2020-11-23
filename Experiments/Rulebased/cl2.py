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

class Runner:
    # consider using factory method to access outer from inner, see e.g.
    # https://stackoverflow.com/questions/2024566/how-to-access-outer-class-from-an-inner-class
    def __init__(self, num_players):
        self.num_players = num_players
        self.environment = rl_env.make('Hanabi-Full', self.num_players)
        self.agent_config = {'players': self.num_players}  # same for all ra.RulebasedAgent instances
        self.zeros = [0 for _ in range(50)]  # used by one hot encoding action function

    def to_one_hot(self, action_dict):
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
        # one_hot_action = [0 for _ in range(50)]
        # one_hot_action[int_action] = 1
        # return one_hot_action
        return int_action

    def _is_target_agent(self, agent_id, agent_cls):
        # if agent_id:
        #     # determine if agent_id corresponds to target agent
        #     keys = list(AGENT_CLASSES.keys())
        #     vals = list(AGENT_CLASSES.values())
        #     try:
        #         return agent_id == keys[vals.index(agent_cls)]
        #     except KeyError as e:
        #         return False
        if agent_id:
            # print(str(agent_cls).__contains__(agent_id))
            # todo little bit crude but shouldnt break as of now
            return str(agent_cls).__contains__(agent_id)
        return True

    def run(self, agents, max_games=1, agent_id=None):
        """
        agents: Agent Classes used to sample players from (uniformly at random for each game)
        max_games: number of games to collect at maximum
        agent_id: if provided, only state-action pairs for corresponding agent are returned
        """

        i_game = 0
        cum_states = []
        cum_actions = []
        turns_played = 0

        while i_game < max_games:
            # create game
            observations = self.environment.reset()
            done = False
            players = agents  # players = [agent(self.agent_config) for agent in agents]

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
                        if self._is_target_agent(agent_cls=agents[agent_index], agent_id=agent_id):
                            # add binary encoding of player position, so that actions offsets
                            # can be understood by the network, use 3 bits to encode (remove this later)
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


class StateActionCollector:
    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 agent_id: Optional[str] = None
                 ):
        self.agent_classes = agent_classes
        self.num_players = num_players
        # self.states = []
        self._agent_id = agent_id
        self.runner = Runner(self.num_players)
        self.initialized_agents = []

    def _initialize_all_agents(self, target_agent):
        """ set self.initialized_agents,
        so that run() calls wont re-initialize them every time
        additionally, return index of target agent, so that it can be fed to run() calls
        target agent is the one we want to collect (s,a) pairs for
        """
        agents = []
        target_index = None
        for i, (agent_id, agent_cls) in enumerate(self.agent_classes.items()):
            agents.append(agent_cls({'players': self.num_players}))
            if target_agent == agent_id:
                target_index = i
        self.initialized_agents = agents
        return target_index

    def get_agents(self, k, target_agent_index=None):
        """ If agent_id is specified, have agent with agent_id at least once in agents list
        This is usually, because we want state,action pairs for this specific agent
        """
        if not target_agent_index:
            agents = []
        else:
            agents = [self.initialized_agents[target_agent_index]]
        # indices = random.choices([i for i in range(len(self.initialized_agents))], k=k)
        # [agents.append(self.initialized_agents[i]) for i in indices]
        agents += random.choices(self.initialized_agents, k=k)
        return agents#, k

    def collect(self, max_states=1000, agent_id=None, games_per_group=1) -> Tuple[List[List], List[List]]:
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
        # initialize all agents only once, and then pass them over to run function
        target_agent_index = self._initialize_all_agents(agent_id)
        k = self.num_players - bool(agent_id)  # maybe save one spot for target agent

        while num_states < max_states:

            # play one game with randomly sampled agents
            agents = self.get_agents(k=k, target_agent_index=target_agent_index)
            states, actions, num_turns_played = self.runner.run(agents, max_games=games_per_group, agent_id=agent_id)

            # cumulated stats
            num_states += num_turns_played
            if not isinstance(cum_states, np.ndarray):
                cum_states = np.array(states)
                cum_actions = np.array(actions)
            else:
                try:
                    cum_states = np.concatenate((cum_states, states))
                    cum_actions = np.concatenate((cum_actions, actions))
                except ValueError:
                    print(states, actions, agents, num_states)
                    exit(1)

        # return random subset of cum_states, cum_actions
        max_len = len(cum_states)
        indices = [random.randint(0,max_len-1) for _ in range(max_states)]
        return cum_states[indices], cum_actions[indices]


