from State import State
from Ball import Ball, Position, Velocity
import copy
import numpy as np


def check_converge(frames):
    """
    Check if the final state has converged
    
    frames: List[State], the frames stocked in order of time
    """
    
    if len(frames) < 200:
        return False
    else:
        last_frames = frames[-200:]
        for frame in last_frames:
            for b in frame.balls:
                if np.linalg.norm(b.velocity) > 0.15:
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
    collision_factor = 0.09  # further tuning needed

    screen_limit = np.array([state.screen_x, state.screen_y])

    dt = 0.01  # time step of evaluation
    t = 0

    frames = [state]  # store the frames of evaluation
    converged = False  

    balls = state.balls
    
    count = 0
    
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
                    b.velocity[j] = 0
                    b.position[j] = b.radius          

                if b.position[j] > screen_limit[j] - b.radius:
                    b.velocity[j] = 0
                    b.position[j] = screen_limit[j] - b.radius

    #         Check collisions between balls
        for i in range(N):
            for j in range(i+1, N):
                ball_1 = balls[i]
                ball_2 = balls[j]
                
                
                if ( np.linalg.norm(ball_1.position - ball_2.position) <= ball_1.radius + ball_2.radius ):
                    
                    m1 = ball_1.radius
                    m2 = ball_2.radius
                    
                    mid_point = (1/(m1 + m2)) * (m2 * ball_1.position +  m1 * ball_2.position)
                    u = (ball_2.position - ball_1.position)  # uniform vector from ball 1 to ball 2
                    u = u / np.linalg.norm(u)
                    #  Update the positions after collision
                    ball_1.position = mid_point - ball_1.radius * u
                    ball_2.position = mid_point + ball_2.radius * u
                    
                    # Update the velocity of balls after collsion
                    # divide the velocity to two dimension : u and n
                    
                    v1_u = np.dot(ball_1.velocity, u) * u
                    v1_n = ball_1.velocity - v1_u
                    
                    v2_u = np.dot(ball_2.velocity, u) * u
                    v2_n = ball_2.velocity - v2_u
                    
                    # the velocity of direction n does not change, but in direction u they exchange

                    
                    v1_u_after = collision_factor * ((m1 - m2) * v1_u + 2 * m2 * v2_u)/(m1 + m2)
                    v2_u_after = collision_factor * ((m2 - m1) * v2_u + 2 * m1 * v1_u)/(m1 + m2)
                    
                    ball_1.velocity = amortize_factor*v1_n + v2_u_after
                    ball_2.velocity = amortize_factor*v2_n + v1_u_after
            
        for i in range(N):
            b = balls[i]
            for j in range(2):
                if b.position[j] < b.radius:
                    b.velocity[j] = 0
                    b.position[j] = b.radius          

                if b.position[j] > screen_limit[j] - b.radius:
                    b.velocity[j] = 0
                    b.position[j] = screen_limit[j] - b.radius
            
        count += 1
        for i in range(N):
            b = balls[i]
            if np.linalg.norm(b.velocity) < 0.05:
                b.velocity = np.array([0,0])
#             if count % 100 == 0:
#                 print(b)
    #       For debug
#         if count % 100 ==
        
        frames.append( copy.deepcopy(state) )
        converged = check_converge(frames)
        t += dt
        if t > 60:  # protection
            break;
    return state