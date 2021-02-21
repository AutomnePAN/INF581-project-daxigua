def test_state_plot():
    from State import State
    from Ball import Ball, Velocity, Position

    balls = []
    ball_nbr = 10
    x = 300
    y = 500
    endline = 0.8
    radius = int(x/ball_nbr/2)
    for i in range(ball_nbr):
        balls.append(Ball(Position(int(x*i/ball_nbr) + radius, radius),
                          Velocity(0, 0),
                          radius=radius,
                          color=(255, 0, 100)))

    state = State(x, y, balls, int(endline*y))
    state.plot_state()