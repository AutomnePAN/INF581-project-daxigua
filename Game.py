from State import State
from Ball import Ball, Position, Velocity
from Movement_evaluation import evaluate_by_gravity

class Game(object):
    
    """
    Implement the environment of the Game
    """
    
    def __init__(self, screen_x, screen_y, balls_setting ):
        """
        screen_x: float, the width of the screen
        screen_y: float, the height of the screen
        balls_setting: Dict( radius: float, score: float, color: tuple ), the sizes of balls and corresponding rewards used in the function
        """
        
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.balls_setting = balls_setting
        self.current_state = None  # TO DO
        self.init_state()
        
#         --TO ADD MORE--
        
    def init_state(self):
#       
        return 

    def check_fin(self):
#         --TO DO--
        return 
    
    def calculate_reward(self):
#         --TO DO--
        return 

#     ---TO ADD MORE---