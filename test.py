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
