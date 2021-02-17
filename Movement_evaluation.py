from State import State
from Ball import Ball, Position, Velocity
import copy
import numpy as np


def check_converge(frames):
    """
    Check if the final state has converged
    
    frames: List[State], the frames stocked in order of time
    """
    
    if len(frames) < 10:
        return False
    else:
        last_frames = frames[-10:]
        for frame in last_frames:
            for b in frame.balls:
                if np.linalg.norm(b.velocity) > 0.01:
                    return False
        return True
    

def evaluate_by_gravity(state):
    """
    state: State, the initial state
    return: State, the final converged state
    
    implement the movement of the balls in the state by the effect of gravity
    """
    
    g = -9.8
    amortize_factor = 0.99  # further tuning needed
    collision_factor = 0.1  # further tuning needed

    screen_limit = np.array([state.screen_x, state.screen_y])

    dt = 0.01  # time step of evaluation
    t = 0

    frames = [state]  # store the frames of evaluation
    converged = False  

    balls = state.balls

    
    while not converged:

        N = len(balls)

        for i in range(N):
            b = balls[i]
            f = np.array([0, 1*g])

    #         Update the velocity
            b.velocity = (1 - amortize_factor * dt) * b.velocity + dt * f

    #         Update the position
            b.position = b.position + b.velocity * dt

    #         Check collision with borders
            for j in range(2):
                if b.position[j] < b.radius:
                    b.velocity[j] = -collision_factor * b.velocity[j]
                    b.position[j] = b.radius          

                if b.position[j] > screen_limit[j] - b.radius:
                    b.velocity[j] = -collision_factor * b.velocity[j]
                    b.position[j] = screen_limit[j] - b.radius

    #         Check collisions between balls
    #         --TO DO--

        for i in range(N):
            b = balls[i]
            if np.linalg.norm(b.velocity) < 0.01:
                b.velocity = np.array([0,0])
    #       For debug
    #         print(b)
        frames.append( copy.deepcopy(state) )
        converged = check_converge(frames)
        t += dt
    return state