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

from State import State
from Movement_evaluation import evaluate_by_gravity
import numpy as np
import random
import time
from Ball import Ball
from Config import *
from Game import Game

class Random_Agent(object):
        
    def get_action(self, state):
        
        return random.randint(0, int(state.screen_x))


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
    # TO DO
    return a_t

