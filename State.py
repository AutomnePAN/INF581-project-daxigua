from Ball import Ball, Position, Velocity

class State(object):
    
    def __init__(self, screen_x, screen_y, balls):
        """
        screen_x: float, the width of the screen
        screen_y: float, the height of the screen
        balls: List[Ball], the  list of balls in the screen
        """
        
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.balls = balls
        
    def plot_state(self):
        """Plot the State with an image"""
        
#         --TO DO--
        return 