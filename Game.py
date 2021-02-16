from State import State
from Ball import Ball, Position, Velocity
from Movement_evalution import evaluate_by_gravity

class Game(object):
    
    """
    Implement the environment of the Game
    """
    
    def __init__(self, screen_x, screen_y, ball_setting ):
        """
        screen_x: float, the width of the screen
        screen_y: float, the height of the screen
        ball_setting: Dict( radius: float, reward: float ), the sizes of balls and corresponding rewards used in the function
        """
        
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.ball_setting = ball_setting
        self.current_state = None  # TO DO
        self.init_state()
        
#         --TO ADD MORE--
        
    def init_state(self):
#         --TO DO--
        return 

    def check_fin(self):
#         --TO DO--
        return 
    
    def calculate_reward():
#         --TO DO--
        return 

#     ---TO ADD MORE---