from Policy_Gradient import *


def test_state_plot():
    from State import State
    from Ball import Ball, Velocity, Position

    ball_nbr = 10
    x = 300
    y = 500
    endline = 0.8
    radius = int(x/ball_nbr/2)
    T = 10
    for t in range(T):
        balls = []
        for i in range(ball_nbr):
            balls.append(Ball(Position(int(x*i/ball_nbr) + radius, radius + t*10),
                              Velocity(0, 0),
                              radius=radius,
                              color=(255, 0, 100)))
        state = State(x, y, int(endline * y), balls=balls)
        state.step = t
        state.balls = balls
        if t == T-1:
            state.is_final = True
        state.plot_state(is_save=True)

def policy_gradient_test():

    gradient_agent = Gradient_Agent(np.zeros((2, 31)))

    final_rewards = []  # sum of the score at each step
    scores = []

    start_time = time.time()
    for i in range(20):
        game = Game(screen_x, screen_y, end_line,
                    balls_setting, max_random_ball_level)
        episode_states, episode_actions, episode_rewards = play_one_episode(
            game, gradient_agent, max_step=100, plot=False)
        print(i, "\t th episode: ", episode_rewards[-1])
        final_rewards.append(episode_rewards[-1])
        scores.append(np.sum(episode_rewards))
        PG = gradient_agent.compute_policy_gradient(
            episode_states, episode_actions, episode_rewards)

        print(PG)

    end_time = time.time()

    print((end_time - start_time) / 60)

def test_train():
    gradient_agent = Gradient_Agent(np.zeros((2, 31)))
    game = Game(screen_x, screen_y, end_line,
                balls_setting, max_random_ball_level)
    theta, episode_index, average_returns = train(game, gradient_agent)
    print(theta)
    print(episode_index)
    print(average_returns)

def test_video():
    import random
    from Game import Game
    from Config import *
    class Random_Agent(object):

        def get_action(self, state):
            return random.randint(0, int(state.screen_x))

    def play_one_episode(game, agent, plot=False, **kwargs):
        is_finish = False
        current_state = game.init_state()
        reward_recorder = []
        step = 1

        while not is_finish:
            action = agent.get_action(current_state)
            next_state, reward, is_finish = game.next_step(action, verbose=False, **kwargs)
            if plot:
                next_state.plot_state()
            reward_recorder.append(reward)
            current_state = next_state

        return reward_recorder[-1], reward_recorder, game.current_reward

    game = Game(screen_x, screen_y, end_line, balls_setting, max_random_ball_level)
    random_agent = Random_Agent()
    play_one_episode(game, random_agent, video=True)

test_train()