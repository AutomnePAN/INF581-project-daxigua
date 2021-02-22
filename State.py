import cv2 as cv
import numpy as np

class State(object):
    
    def __init__(self, screen_x, screen_y, balls, endline, bg_color=(200, 210, 221)):
        """
        screen_x: float, the width of the screen
        screen_y: float, the height of the screen
        balls: List[Ball], the  list of balls in the screen
        """
        self.screen_x = screen_x
        self.screen_y = screen_y
        assert endline < screen_y
        self.endline = endline
        self.balls = balls
        self.bg_color = bg_color
        self.canvas = InitCanvas(self.screen_x, self.screen_y, color=bg_color)

    def reset_canvas(self):
        self.canvas = InitCanvas(self.screen_x, self.screen_y, color=self.bg_color)

    def plot_state(self):
        """Plot the State with an image"""
        self.reset_canvas()
        cv.line(self.canvas,
                (0, self.screen_y - self.endline),
                (self.screen_x, self.screen_y - self.endline),
                (0, 0, 255), thickness=2)
        for ball in self.balls:
            print(ball)
            cv.circle(self.canvas,
                      (int(ball.position[0]), int(self.screen_y-ball.position[1]) ),
                      int(ball.radius), ball.color, -1)
        print(64 * "-")
        cv.imshow('state', self.canvas)
        cv.waitKey(0)
        cv.destroyWindow('state')


def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas