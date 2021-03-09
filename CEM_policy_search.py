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

import sys, subprocess
import time

import itertools


from State import State
from Movement_evaluation import evaluate_by_gravity
import numpy as np
import random
import time
from Ball import Ball
from Config import *
from Game import Game

def play_one_episode(game, agent, max_step = None, plot = False):
    is_finish = False
    current_state = game.init_state()
    reward_recorder = []
    step = 1

    while not is_finish:
        action = agent.get_action(current_state)
        next_state, reward, is_finish = game.next_step(action, verbose = False)
        if plot:
            next_state.plot_state()
        reward_recorder.append(reward)
        current_state = next_state
        step += 1
        if max_step and step >= max_step:
            break
    
    return reward_recorder[-1], reward_recorder, game.current_reward
    
def sigmoid(x):
    """
    x: float
    """
    return 1.0 / (1.0 + np.exp(-x))


def logistic_regression(s, theta):
    """
    s: State
    theta: np.array, with the same dimension as the vectorized state
    """
    left2right_proportion = sigmoid(np.dot(s.vectorize(), np.transpose(theta)))
    return left2right_proportion


def draw_action(s, theta):
    return int(s.screen_x * logistic_regression(s, theta))
    

class LogisticRegression:

    def __init__(self, game):        
        self.num_params = len(game.current_state.vectorize())

    def __call__(self, state, theta):
        return draw_action(state, theta)
    

class LogisticRegressionAgent(object):
    
    def __init__(self, theta):
        self.theta = theta
    
    def get_action(self, state):
        return draw_action(state, self.theta)
        
class ObjectiveFunction:

    def __init__(self, game, policy, num_episodes=1, max_time_steps=float('inf'), minimization_solver=True):
        self.ndim = policy.num_params  # Number of dimensions of the parameter (weights) space
        self.game = game
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimization_solver = minimization_solver

        self.num_evals = 0

        
    def eval(self, policy_params, num_episodes=None, max_time_steps=None, render=False):
        """Evaluate a policy"""

        self.num_evals += 1

        if num_episodes is None:
            num_episodes = self.num_episodes

        if max_time_steps is None:
            max_time_steps = self.max_time_steps

        average_total_rewards = 0

        for i_episode in range(num_episodes):

            total_rewards = 0.
            state = self.game.init_state()
            self.game.is_finish = False
            for t in range(max_time_steps):
                if render:
                    self.game.current_state.plot_state(is_plt=True)

                action = self.policy(state, policy_params)
#                 print(action)
                state, reward, is_finish = self.game.next_step(action, verbose = False)
                total_rewards = reward
                
                if is_finish:
                    break

            average_total_rewards += float(total_rewards) / num_episodes

            if render:
                print("Test Episode {0}: Total Reward = {1}".format(i_episode, total_rewards))

        if self.minimization_solver:
            average_total_rewards *= -1.

        return average_total_rewards   # Optimizers do minimization by default...

    
    def __call__(self, policy_params, num_episodes=None, max_time_steps=None, render=False):
        return self.eval(policy_params, num_episodes, max_time_steps, render)
        

def cem_uncorrelated(objective_function,
                     mean_array,
                     var_array,
                     max_iterations=500,
                     sample_size=30,
                     elite_frac=0.2,
                     print_every=10,
                     success_score=float("inf"),
                     num_evals_for_stop=None,
                     hist_dict=None):
    """Cross-entropy method.
    Params
    ======
        objective_function (function): the function to maximize
        mean_array (array of floats): the initial proposal distribution (mean vector)
        var_array (array of floats): the initial proposal distribution (variance vector)
        max_iterations (int): number of training iterations
        sample_size (int): size of population at each iteration
        elite_frac (float): rate of top performers to use in update with elite_frac ∈ ]0;1]
        print_every (int): how often to print average score
        hist_dict (dict): logs
    """
    assert 0. < elite_frac <= 1.
    n_elite = math.ceil(sample_size * elite_frac)
    optimal_score = 100
    theta = mean_array
    
    for iteration_index in range(0, max_iterations):
        # SAMPLE A NEW POPULATION OF SOLUTIONS (THETA VECTORS) ################
        theta_array = np.random.multivariate_normal(mean=mean_array, cov=np.diag(var_array), size=(sample_size))
        # EVALUATE SAMPLES AND EXTRACT THE BEST ONES ("ELITE") ################
        score_array = np.array([objective_function(theta) for theta in theta_array])
        sorted_indices_array = score_array.argsort()             
        elite_indices_array = sorted_indices_array[:n_elite]     
        elite_theta_array = theta_array[elite_indices_array]
        # FIT THE NORMAL DISTRIBUTION ON THE ELITE POPULATION #################
        mean_array = elite_theta_array.mean(axis=0)
        var_array = elite_theta_array.var(axis=0)
        score = objective_function(mean_array)
        if iteration_index % print_every == 0:
            print("Iteration {}\tScore {}".format(iteration_index, score))
#             print("mean array: ", mean_array)
            print("var array max: ", np.max(np.abs(var_array)))
            print("scores: ", score_array)
            print("elite indices", elite_indices_array)
        if hist_dict is not None:
            hist_dict[iteration_index] = [score] + mean_array.tolist() + var_array.tolist()
        if num_evals_for_stop is not None:
            score = objective_function(mean_array, num_episodes=num_evals_for_stop)
        
        if score < optimal_score:
            optimal_score = score
            np.save( "theta.npy", mean_array )
            theta= mean_array
            
        if score <= success_score:
            break
            
        if np.max(np.abs(var_array)) < 10 ** -2:
            break
    return theta
    
    

# Activation functions ########################################################

def identity(x):
    return x

def tanh(x):
    return np.tanh(x)

def relu(x):
    x_and_zeros = np.array([x, np.zeros(x.shape)])
    return np.max(x_and_zeros, axis=0)

# Dense Multi-Layer Neural Network ############################################

class NeuralNetworkPolicy:

    def __init__(self, game, theta=None, h_size=16):   # h_size = number of neurons on the hidden layer
        # Set the neural network activation functions (one function per layer)
        self.activation_functions = (relu, relu, sigmoid)
        
        # Make a neural network with 1 hidden layer of `h_size` units
        weights = (np.zeros([len(game.current_state.vectorize()) + 1, h_size]),
                   np.zeros([h_size + 1, h_size]),
                   np.zeros([h_size + 1, 1]))

        self.shape_list = weights_shape(weights)
        print("Number of parameters per layer:", self.shape_list)
        
        self.num_params = len(flatten_weights(weights))
        print("Number of parameters (neural network weights) to optimize:", self.num_params)
        
        self.theta = theta

    def __call__(self, state, theta):
        weights = unflatten_weights(theta, self.shape_list)

        return feed_forward(inputs=state,
                            weights=weights,
                            activation_functions=self.activation_functions)
    
    def get_action(self, state):
        return self.__call__(state, self.theta)
        
def feed_forward(inputs, weights, activation_functions, verbose=False):

    state = inputs
    inputs = inputs.vectorize()
    hidden_layer_1 = activation_functions[0]( np.dot( np.append(inputs, [1]), weights[0] ) )
    hidden_layer_2 = activation_functions[1]( np.dot( np.append(hidden_layer_1, [1]), weights[1] ) )
    layer_output = activation_functions[2]( np.dot( np.append(hidden_layer_2, [1]), weights[2] ) )
    if verbose:
          print("weight: \t", layer_output)
#     print(layer_output)
    return int( state.screen_x * layer_output)


def weights_shape(weights):
    return [weights_array.shape for weights_array in weights]


def flatten_weights(weights):
    """Convert weight parameters to a 1 dimension array (more convenient for optimization algorithms)"""
    nested_list = [weights_2d_array.flatten().tolist() for weights_2d_array in weights]
    flat_list = list(itertools.chain(*nested_list))
    return flat_list


def unflatten_weights(flat_list, shape_list):
    """The reverse function of `flatten_weights`"""
    length_list = [shape[0] * shape[1] for shape in shape_list]

    nested_list = []
    start_index = 0

    for length, shape in zip(length_list, shape_list):
        nested_list.append(np.array(flat_list[start_index:start_index+length]).reshape(shape))
        start_index += length

    return nested_list
    
    