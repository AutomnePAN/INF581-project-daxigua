import cv2 as cv
import numpy as np

class State(object):

    def __init__(self,
                 screen_x,
                 screen_y,
                 endline,
                 balls,
                 score=0,
                 is_begin=True,
                 is_final=False,
                 bg_color=(200, 210, 221)):
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
        self.score = score
        self.is_begin = is_begin
        self.is_final = is_final
        self.bg_color = bg_color
        self.canvas = InitCanvas(self.screen_x, self.screen_y, color=self.bg_color)
        self.step = 0

    def plot_state(self, is_save=False, path="./result/", speed=300):
        """Plot the State with an image"""
        if not self.is_begin:
            self.is_begin = True
            cv.namedWindow('state', cv.WINDOW_NORMAL)
        cv.line(self.canvas,
                (0, self.screen_y - self.endline),
                (self.screen_x, self.screen_y - self.endline),
                (0, 0, 255), thickness=2)
        cv.putText(self.canvas, "Score : {}".format(self.score), (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        for ball in self.balls:
            print(ball)
            cv.circle(self.canvas,
                      (int(ball.position[0]), int(self.screen_y-ball.position[1]) ),
                      int(ball.radius), ball.color, -1)
        print(64 * "-")
        cv.imshow('state', self.canvas)
        if is_save:
            cv.imwrite((path+"{}.jpg").format(self.step), img=self.canvas)
            print("saved")
        if self.is_final:
            cv.waitKey(0)
            cv.destroyWindow('state')
        cv.waitKey(speed)

        return self.canvas

    def video(self):
        """TODO: Save the video"""
        pass


def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas