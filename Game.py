from State import State
from Ball import Ball, Position, Velocity
from Movement_evaluation import evaluate_by_gravity
import random
import numpy as np

class Game(object):

    """
    Implement the environment of the Game
    """
    
    def __init__(self, screen_x, screen_y, end_line, balls_setting, max_random_ball_level):
        """
        screen_x: float, the width of the screen
        screen_y: float, the height of the screen
        end_line: the line to indicate the failure of the game
        balls_setting: Dict( radius: float, score: float, color: tuple ), the sizes of balls and corresponding rewards used in the function
        max_random_ball_level: integer, the max level of the ball to add at each step
        """

        self.screen_x = screen_x
        self.screen_y = screen_y
        self.end_line = end_line
        self.max_random_ball_level = max_random_ball_level
        self.balls_setting = balls_setting
        self.current_state = None
        self.current_reward = 0
        self.is_finish = False
        self.init_state()
        
    def init_state(self):
        """Initialize the state with one ball at the middle top of the canvas"""
        balls = [self.random_new_ball()]
        state = State(self.screen_x, self.screen_y, balls, self.end_line)
        self.current_state = state
        self.current_reward = 0
        return state

    def random_new_ball(self):
        """We only random a ball with level between [0, max_ball_level]"""
        ball_level = random.randint(0, self.max_random_ball_level)
        # Place the ball at the top middle of the canvas
        pos = np.array([self.screen_x/2, (self.screen_y + self.end_line)/2])
        # Set the initial velocity as 0
        vel = np.array([0, 0])
        new_ball = Ball(pos, vel, ball_level)
        return new_ball

    def next_step(self, action, verbose = True):
        """
        action: float, the x position of the new ball to drop with
        """
        if self.is_finish:
            print('The game is finish.')
            return
        # Move the latest ball in the current state to the x_position indicated by action
        self.current_state.balls[0].position[0] = action
        self.current_state, obtained_score = evaluate_by_gravity(self.current_state, plot=False, verbose= verbose)
        self.current_reward += obtained_score

        self.is_finish = self.check_fin()

        if not self.is_finish:
            # Add a new ball into the state
            self.current_state.balls.append(self.random_new_ball())
        else:
            # Add the score corresponding to all the balls created
            final_step_score = 0
            for ball in self.current_state.balls:
                final_step_score += self.balls_setting[ball.ball_level]['score']
            self.current_reward += final_step_score
            print('The game is finish, final score is {}'.format(self.current_reward))

        return self.current_state, self.current_reward, self.is_finish


    def check_fin(self):
        """Check if all the balls are under the endline."""
        for ball in self.current_state.balls:
            if ball.position[1] + ball.radius > self.end_line:
                return True
        return False
