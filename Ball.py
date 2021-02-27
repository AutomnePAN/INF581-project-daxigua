import numpy as np
from Config import balls_setting

class Position(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Velocity(object):
    def __init__(self, vx, vy):
        self.x = vx
        self.y = vy
        
class Ball(object):
    def __init__(self, position, velocity, ball_type):
        '''
        position: numpy.array, the center of ball
        velocity: numpy.array, the velocity vector of ball
        balls_setting: Dict( radius: float, score: float, color: tuple ), the sizes of balls and corresponding rewards used in the game
        ball_type: integer, index of the type of the created ball in the balls_setting
        radius: int, radius of ball
        color: tuple, RGB value of ball
         '''
        self.position = position
        self.velocity = velocity
        self.ball_type = ball_type
        self.radius = balls_setting[ball_type]['radius']
        self.color = balls_setting[ball_type]['color']

    def __str__(self):
        return f"Position, x: {self.position[0]}, y: {self.position[1]} \nVelocity, vx: {self.velocity[0]}, vy: {self.velocity[1]} \nRadius: {self.radius}\n"

    def change_ball_type(self, new_ball_type):
        # change the type of the ball
        self.radius = balls_setting[new_ball_type]['radius']
        self.color = balls_setting[new_ball_type]['color']


