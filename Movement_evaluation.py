from State import State
from Ball import Ball, Position, Velocity


def check_converge(frames):
    """
    Check if the final state has converged
    
    frames: List[State], the frames stocked in order of time
    """
    
#     --TO DO--
    return True


def evaluate_by_gravity(state):
    """
    state: State, the initial state
    return: State, the final converged state
    
    implement the movement of the balls in the state by the effect of gravity
    """
    
    g = 9.8
    amortize_factor = 0.1  # further tuning needed
    collision_factor = 0.1  # further tuning needed
    dt = 0.1  # time step of evaluation
    t = 0
    
    frames = [state]  # store the frames of evaluation
    converged = False  
    
    while not converged:
        
#         Check the collision
#         --TO DO--
#         Check the gravity effect
#         --TO DO--
        converged = check_converge(frames)
        t += dt
    return state