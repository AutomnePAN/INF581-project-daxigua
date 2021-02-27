from State import State
from Ball import Ball, Position, Velocity
from Movement_evaluation import evaluate_by_gravity
import random

class Game(object):
    
    """
    Implement the environment of the Game
    """
    
    def __init__(self, screen_x, screen_y, end_line, balls_setting):
        """
        screen_x: float, the width of the screen
        screen_y: float, the height of the screen
        balls_setting: Dict( radius: float, score: float, color: tuple ), the sizes of balls and corresponding rewards used in the function
        max_random_ball_level: integer, the max level of the ball to add at each step
        """
        
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.end_line = end_line
        self.balls_setting = balls_setting
        self.current_state = None  # TO DO
        self.init_state()
        
#         --TO ADD MORE--
        
    def init_state(self):
#
        return

    def random_new_ball(self, max_random_ball_level):
        """we only random a ball with level between [0, max_ball_level]"""
        ball_type = random.randint(0, max_random_ball_level)


    def check_fin(self):
#         --TO DO--
        return 
    
    def calculate_reward(self):
#         --TO DO--
        return 

#     ---TO ADD MORE---