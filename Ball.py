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
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.color = color
    
    def __str__(self):
        return f"Position, x: {self.position.x}, y: {self.position.y} \nVelocity, vx: {self.velocity.x}, vy: {self.velocity.y} \nRadius: {self.radius}"