import copy
import math
import random
import subprocess
import sys
import time
from abc import abstractmethod
from os import name
from typing import List

import gym
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image
from sklearn.preprocessing import normalize

from Ball import Ball
from Config import *
from Game import Game
from Movement_evaluation import evaluate_by_gravity
from State import State


def sigmoid(x: float):
    """
    x: float
    """
    return 1.0 / (1.0 + np.exp(-x))


class Agent(object):
    @abstractmethod
    def get_action(self, state: State):
        pass


class Random_Agent(Agent):
    def get_action(self, state: State):
        return random.randint(0, int(state.screen_x))


class Gradient_Agent(Agent):

    def __init__(self, theta) -> None:
        super().__init__()
        self.theta = theta

    def logistic_regression(self, s: State):
        mu, sigma = np.dot(s.vectorize(), np.transpose(self.theta))
        return s.screen_x * sigmoid(mu), np.exp(sigma)

    def get_action(self, s: State):
        mu, sigma = self.logistic_regression(s)
        #print("mu: {} \t sigma: {}".format(mu, sigma))
        return np.clip(np.random.normal(mu, sigma, 1), 0, s.screen_x)

    def compute_policy_gradient(self, episode_states: List[State], episode_actions: List[float], episode_rewards: List[int]):

        H = len(episode_rewards)
        PG = 0

        for t in range(H):
            mu, sigma = self.logistic_regression(episode_states[t])
            s_t = episode_states[t].vectorize()
            a_t = episode_actions[t]
            R_t = sum(episode_rewards[t::])
            g_gaussian = [(a_t-mu)/(sigma**2),
                          ((a_t-mu)**2-sigma**2)/(sigma**3)]
            mu_partiel_theta = np.array([s_t, np.zeros(78)])
            sigma_partiel_theta = np.array([np.zeros(78), s_t])
            g_theta_log_pi = g_gaussian[0]*mu_partiel_theta*mu*(1-mu/episode_states[t].screen_x) + \
                g_gaussian[1]*sigma_partiel_theta*sigma
            PG += g_theta_log_pi * R_t

        return PG


def play_one_episode(game: Game, agent: Agent, max_step=None, plot=False):
    is_finish = False
    current_state = game.init_state()
    episode_states = []
    episode_actions = []
    episode_rewards = []

    episode_states.append(current_state)
    step = 1

    while not is_finish:
        action = agent.get_action(current_state)
        # print("action: " + str(action))
        next_state, reward, is_finish = game.next_step(action, verbose=False)
        if plot:
            next_state.plot_state()
        current_state = next_state
        if episode_rewards == []:
            episode_rewards.append(reward)
        else:
            episode_rewards.append(reward-np.sum(episode_rewards))
        episode_actions.append(action)
        episode_states.append(current_state)
        step += 1
        # print(episode_rewards)
        if max_step and step >= max_step:
            break
    print(episode_rewards)
    print("")
    return episode_states, episode_actions, episode_rewards


def score_on_multiple_episodes(game: Game, agent, score=SCORE, num_episodes=NUM_EPISODES, plot=False):

    num_success = 0
    average_return = 0
    num_consecutive_success = [0]
    rewards = []

    for episode_index in range(num_episodes):
        _, _, episode_rewards = play_one_episode(
            game, agent)

        total_rewards = sum(episode_rewards)

        if total_rewards >= score:
            num_success += 1
            num_consecutive_success[-1] += 1
        else:
            num_consecutive_success.append(0)

        average_return += (1.0 / num_episodes) * total_rewards

        if plot:
            print("Test Episode {0}: Total Reward = {1} - Success = {2}".format(
                episode_index+1, total_rewards, total_rewards > score))
        rewards.append(total_rewards)

    # MAY BE ADAPTED TO SPEED UP THE LERNING PROCEDURE
    if max(num_consecutive_success) >= NUM_CONSECUTIVE_SUCCESS:
        success = True
    else:
        success = False

    return success, num_success, average_return, rewards


def train(game: Game, agent: Gradient_Agent, alpha_init=ALPHA_INIT):

    episode_index = 0
    average_returns = []

    success, _, R, _ = score_on_multiple_episodes(game, agent, plot=True)
    average_returns.append(R)

    # Train until success
    # while (not success):
    for i in range(500):

        # Rollout
        episode_states, episode_actions, episode_rewards = play_one_episode(
            game, agent)

        # Schedule step size
        # alpha = alpha_init
        alpha = alpha_init / (1 + 0.1*episode_index)

        # Compute gradient
        PG = agent.compute_policy_gradient(
            episode_states, episode_actions, episode_rewards)
        # print variable value
        # print(theta)
        # restrict the range of gradient to avoid gradient explosion

        PG = normalize(PG)
        # print(PG)

        # Do gradient ascent
        agent.theta += alpha * PG

        # Test new policy
        success, _, R, _ = score_on_multiple_episodes(
            game, agent, plot=True)

        # Monitoring
        average_returns.append(R)

        episode_index += 1

        print("Episode {0}, average return: {1}".format(episode_index, R))

    return agent.theta, episode_index, average_returns


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
