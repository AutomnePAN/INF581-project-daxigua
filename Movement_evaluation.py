from State import State
from Ball import Ball, Position, Velocity
import copy
import numpy as np
from Config import balls_setting

def check_converge(frames, tolerance=5):
    """
    Check if the final state has converged
    
    frames: List[State], the frames stocked in order of time
    """
    
    last_frames = frames[-10:]
    current_frame = last_frames[-1]
    for frame in last_frames:
        if len(frame.balls) != len(current_frame.balls):
            return False

        for b1, b2 in zip(frame.balls, current_frame.balls):
#             print(np.linalg.norm(b1.position - b2.position), b1.position, b2.position)
            if np.linalg.norm(b1.position - b2.position) > tolerance:
                return False
    return True
    

def evaluate_by_gravity(state, plot=False, dt=0.1, check_converge_step = 10, protection_time_limit = 30, verbose= False):
    """
    state: State, the initial state
    plot: bool, if plot the progress of the movement

    return:
        State: the final converged state
        obtained_score: score obtained during the evaluation
    
    implement the movement of the balls in the state by the effect of gravity
    """
    
    g = -39.8
    amortize_factor = 1.5  # further tuning needed
    collision_factor = 0.5  # further tuning needed

    screen_limit = np.array([state.screen_x, state.screen_y])

    t = 0

    frames = [state]  # store the frames of evaluation
    converged = False  

    balls = state.balls

    obtained_score = 0
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
                    
                if j == 0 and b.position[j] > screen_limit[j] - b.radius:
                    b.velocity[j] = 0
                    b.position[j] = screen_limit[j] - b.radius

    #         Check collisions between balls
        i = 0
        while i < len(balls):
            j = i + 1
            while j < len(balls):
                ball_1 = balls[i]
                ball_2 = balls[j]
                
                if ( np.linalg.norm(ball_1.position - ball_2.position) <= ball_1.radius + ball_2.radius ):
                    
                    m1 = ball_1.radius
                    m2 = ball_2.radius
                    # collision happens if the two balls have different types or the two balls are on the highest level
                    if m1 != m2 or (m1 == m2 and ball_1.ball_level == max(balls_setting)):
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
                        j += 1
                  
                    else:
                        #  Obtain score when remove one ball
                        removed_ball_level = balls_setting[balls[j].ball_level]
                        obtained_score += removed_ball_level['score']
                        if verbose:
                            print('Remove a {}, obtain {} score'.format(removed_ball_level['name'], removed_ball_level['score']))

                        #  Upgrade the two same level ball into one ball of a higher level
                        mid_point = (1/(m1 + m2)) * (m2 * ball_1.position +  m1 * ball_2.position)


                        del balls[j]
                        balls[i].position = mid_point
                        balls[i].velocity = np.array([0, 0])
                        # Upgrade the ball into next level
                        balls[i].change_ball_level(balls[i].ball_level + 1)

                else:
                    j += 1
            i += 1
     
        for i in range(len(balls)):
            b = balls[i]
            for j in range(2):
                if b.position[j] < b.radius:
                    b.velocity[j] = 0
                    b.position[j] = b.radius          

                if j == 0 and b.position[j] > screen_limit[j] - b.radius:
                    b.velocity[j] = 0
                    b.position[j] = screen_limit[j] - b.radius
            
        count += 1
        for i in range(len(balls)):
            b = balls[i]
            if np.linalg.norm(b.velocity) < 0.5 * -g * dt:
                b.velocity = np.array([0,0])

        if plot:
            if count % 5 == 0:
                state.plot_state()
        
        frames.append( copy.deepcopy(state) )
        if len(frames) >= check_converge_step and len(frames) % check_converge_step == 0:
            converged = check_converge(frames)
        t += dt
        if t > protection_time_limit:  # protection, need more tuning
            break;
    return state, obtained_score