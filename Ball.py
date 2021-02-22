import numpy

class Position(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Velocity(object):
    def __init__(self, vx, vy):
        self.x = vx
        self.y = vy
        
class Ball(object):
    def __init__(self, position, velocity, radius, color=(255, 255, 255)):
        '''
        position: numpy.array, the center of ball
        velocity: numpy.array, the velocity vector of ball
        radius: int, radius of ball
        color: tuple, RGB value of ball
         '''
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.color = color
    
    def __str__(self):
        return f"Position, x: {self.position[0]}, y: {self.position[1]} \nVelocity, vx: {self.velocity[0]}, vy: {self.velocity[1]} \nRadius: {self.radius}\n"