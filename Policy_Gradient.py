from abc import abstractmethod
from os import name
import matplotlib
import matplotlib.pyplot as plt

import math
import gym
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import time

import imageio

import IPython
from IPython.display import Image

import sys
import subprocess

from State import State
from Movement_evaluation import evaluate_by_gravity
import numpy as np
import random
import time
from Ball import Ball
from Config import *
from Game import Game
from typing import List


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
        return mu, np.exp(sigma)

    def get_action(self, s: State):
        mu, sigma = self.logistic_regression(s)
        return np.clip(np.random.normal(mu, sigma, 1), 0, s.screen_x)

    def compute_policy_gradient(self, episode_states: List[State], episode_actions: List[float], episode_rewards: List[int]):

        H = len(episode_rewards)
        PG = 0
        for t in range(H):

            mu, sigma = logistic_regression(episode_states[t], self.theta)
            a_t = episode_actions[t]
            R_t = sum(episode_rewards[t::])
            s_t = episode_states[t].vectorize().reshape(-1, 1)
            n_s_t, _ = s_t.shape
            g_gaussian = [(a_t-mu)/(sigma**2),
                          ((a_t-mu)**2-sigma**2)/(sigma**3)]
            g_theta_log_pi = np.hstack([s_t, np.zeros(
                (n_s_t, 1))])*g_gaussian[0] + np.hstack([np.zeros((n_s_t, 1)), s_t])*g_gaussian[1]
            PG += g_theta_log_pi * R_t

        return PG

    # Train the agent got an average reward greater or equals to 195 over 100 consecutive trials


def sigmoid(x: float):
    """
    x: float
    """
    return 1.0 / (1.0 + np.exp(-x))


def logistic_regression(s: State, theta: np.array):
    """
    s: State
    theta: np.array, with the same dimension as the vectorized state

    Example:
    >>> logistic_regression(Game(screen_x, screen_y, end_line, balls_setting, max_random_ball_level).init_state(), np.zeros((2, 31)))
    (0.0, 1.0)
    """

    mu, sigma = np.dot(s.vectorize(), np.transpose(theta))
    return mu, np.exp(sigma)


def draw_action(s: State, theta):
    mu, sigma = logistic_regression(s, theta)
    return np.clip(np.random.normal(mu, sigma, 1), 0, s.screen_x)


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
        print("action: " + str(action))
        next_state, reward, is_finish = game.next_step(action, verbose=True)
        if plot:
            next_state.plot_state()
        current_state = next_state
        episode_rewards.append(reward)
        episode_actions.append(action)
        episode_states.append(current_state)
        step += 1
        if max_step and step >= max_step:
            break

    print("HHHHHHHH")
    return episode_states, episode_actions, episode_rewards


# Returns Policy Gradient for a given episode
def compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta):

    H = len(episode_rewards)
    PG = 0

    for t in range(H):

        mu, sigma = logistic_regression(episode_states[t], theta)
        a_t = episode_actions[t]
        R_t = sum(episode_rewards[t::])
        g_gaussian = [(a_t-mu)/(sigma**2), ((a_t-mu)**2-sigma**2)/(sigma**3)]
        g_theta_log_pi = np.dot(episode_states[t], g_gaussian)
        PG += g_theta_log_pi * R_t

    return PG


def score_on_multiple_episodes(game: Game, agent, score=SCORE, num_episodes=NUM_EPISODES, plot=False):

    num_success = 0
    average_return = 0
    num_consecutive_success = [0]

    for episode_index in range(num_episodes):
        _, _, episode_rewards = play_one_episode(
            game, agent, plot=plot)

        total_rewards = sum(episode_rewards)

        if total_rewards >= score:
            num_success += 1
            num_consecutive_success[-1] += 1
        else:
            num_consecutive_success.append(0)

        average_return += (1.0 / num_episodes) * total_rewards

        if plot:
            print("Test Episode {0}: Total Reward = {1} - Success = {2}".format(
                episode_index, total_rewards, total_rewards > score))

    # MAY BE ADAPTED TO SPEED UP THE LERNING PROCEDURE
    if max(num_consecutive_success) >= NUM_CONSECUTIVE_SUCCESS:
        success = True
    else:
        success = False

    return success, num_success, average_return


def train(game: Game, agent: Gradient_Agent, alpha_init=ALPHA_INIT):

    theta = agent.theta
    episode_index = 0
    average_returns = []

    success, _, R = score_on_multiple_episodes(game, agent)
    average_returns.append(R)

    # Train until success
    while (not success):

        # Rollout
        episode_states, episode_actions, episode_rewards = play_one_episode(
            game, agent)

        # Schedule step size
        #alpha = alpha_init
        alpha = alpha_init / (1 + episode_index)

        # Compute gradient
        PG = compute_policy_gradient(
            episode_states, episode_actions, episode_rewards, theta)

        # Do gradient ascent
        theta += alpha * PG
        agent.theta = theta

        # Test new policy
        success, _, R = score_on_multiple_episodes(
            game, agent, plot=False)

        # Monitoring
        average_returns.append(R)

        episode_index += 1

        print("Episode {0}, average return: {1}".format(episode_index, R))

    return theta, episode_index, average_returns


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
