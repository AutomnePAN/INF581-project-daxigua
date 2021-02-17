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
    
    def __init__(self, position, velocity, radius):
        """
        position: numpy array
        velocity: numpy array
        """
        self.position = position
        self.velocity = velocity
        self.radius = radius
    
    def __str__(self):
        return f"Position, x: {self.position[0]}, y: {self.position[1]} \nVelocity, vx: {self.velocity[0]}, vy: {self.velocity[1]} \nRadius: {self.radius}"