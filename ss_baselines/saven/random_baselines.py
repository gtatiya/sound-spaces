import json
from collections import defaultdict
from logging import Logger
from pickle import STOP

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
        "RandomAgentWithoutStop", "RandomAgentWithStop"
    ], "EVAL.NONLEARNING.AGENT must be either RandomAgentWithoutStop or RandomAgentWithStop."

    if config.EVAL.NONLEARNING.AGENT == "RandomAgentWithoutStop":
        agent = RandomAgentWithoutStop()
    else:
        agent = RandomAgentWithStop()

    eps = 1e-5
    stop_radius = config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE + eps
    
    logger.info(f"Success radius: {stop_radius}")

    stats = defaultdict(float)
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        goal = np.array(env.current_episode.goals[0].position)
        agent.reset()
        actions = 0
        
        while not env.episode_over:
            action = agent.act(obs)
            actions += 1
            obs = env.step(action)
            curr_pos = np.array(env.sim.get_agent_state().position)
            radius = np.linalg.norm(curr_pos-goal)
            if radius <= stop_radius:
                logger.info(f"agent stopped after {actions} actions and radius: {radius}")
                obs = env.step(HabitatSimActions.STOP)
                
        for m, v in env.get_metrics().items():
            stats[m] += v

    stats = {k: v / num_episodes for k, v in stats.items()}
    
    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)

class RandomAgentWithoutStop(Agent):
    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = np.ones(shape=len(self.actions)) / len(self.actions)
        
        print(f"Random agent intialized with probs: {self.probs}")

    def reset(self):
        pass

    def act(self, observations):
        return {
            "action": np.random.choice(self.actions, p=self.probs)
        }
        
class RandomAgentWithStop(Agent):
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
            self.probs = [0.001, 0.333, 0.333, 0.333]
        
        print(f"Random agent intialized with probs: {self.probs}")

    def reset(self):
        pass

    def act(self, observations):
        return {
            "action": np.random.choice(self.actions, p=self.probs)
        }