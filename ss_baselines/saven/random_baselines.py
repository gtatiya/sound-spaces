import json
from collections import defaultdict

import numpy as np
import habitat
from habitat_baselines.common.baseline_registry import baseline_registry
from typing import Optional

from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm

from habitat import Config, Dataset
from ss_baselines.common.environments import SAVENInferenceEnv

def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    
    config.defrost()
    
    # turn off RGBD rendering as the random agents dont use it
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.freeze()
    
    env = Env(config=config.TASK_CONFIG)

    assert config.EVAL.NONLEARNING.AGENT in [
        "RandomAgent",
    ], "EVAL.NONLEARNING.AGENT must be either RandomAgent or UniformRandomAgent."

    if config.EVAL.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    else:
        agent = UniformRandomAgent()

    stats = defaultdict(float)
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        print(f'observations:\n{obs}')
        
        break
    #     agent.reset()

    #     while not env.episode_over:
    #         action = agent.act(obs)
    #         obs = env.step(action)

    #     for m, v in env.get_metrics().items():
    #         stats[m] += v

    # stats = {k: v / num_episodes for k, v in stats.items()}

    # logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    # for stat_key in stats.keys():
    #     logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    # with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
    #     json.dump(stats, f, indent=4)

class RandomAgent(Agent):
    r"""Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [0.02, 0.68, 0.15, 0.15]

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self.actions, p=self.probs)}


class UniformRandomAgent(Agent):
    r"""Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [0.02, 0.68, 0.15, 0.15]

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self.actions, p=self.probs)}
