import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class State(object):

    def __init__(self,
                 screen_x,
                 screen_y,
                 balls,
                 endline,
                 score=0,
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
        self.is_begin = True
        # self.is_finish = False
        self.bg_color = bg_color
        self.step = 0
        self.canvas = InitCanvas(self.screen_x, self.screen_y, color=self.bg_color)

    def vectorize(self, N = 10):
        """
        vectorize current state by a vector of dimension 3*N + 1
        We suppose there is only one ball above the endline
        """
        result = np.zeros(3 * N + 1)
        self.balls.sort(key = lambda b: b.position[1], reverse=True)
        k = 0
        for i in range(min(N, len(self.balls))):
            if self.balls[i].position[1] > self.endline:
                result[0] = self.balls[i].radius
            else:
                result[3 * k + 1 : 3 * k + 3] = self.balls[i].position
                result[3 * k + 3] = self.balls[i].radius
                k += 1
        return result
        
    
    def plot_state(self, is_save=True, path="./result/", is_plt=False):
        """Plot the State with an image"""
        if self.is_begin:
            self.is_begin = False
            # cv.namedWindow('state', cv.WINDOW_NORMAL)
            self.canvas = cv.line(self.canvas,
                    (0, self.screen_y - self.endline),
                    (self.screen_x, self.screen_y - self.endline),
                    (0, 0, 255), thickness=2)

        cur_canvas = self.canvas.copy()
        for ball in self.balls:
            cv.circle(cur_canvas,
                      (int(ball.position[0]), int(self.screen_y-ball.position[1])),
                      int(ball.radius), ball.color, -1)

            text = f"{ball.ball_level + 1}"
            textsize = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv.putText(cur_canvas,
                       text,
                       (int(ball.position[0]-textsize[0]/2), int(self.screen_y-ball.position[1]+textsize[1]/2)),
                       cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv.putText(cur_canvas, "Score : {}".format(self.score), (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

        if is_save:
            folder = os.path.exists(path)
            if not folder:
                os.makedirs(path)
            cv.imwrite((path+"{}.jpg").format(self.step), img=cur_canvas)

        if is_plt:
            plt.imshow(cv.cvtColor(cur_canvas, cv.COLOR_BGR2RGB))
            plt.xticks([])
            plt.yticks([])
            plt.show()
        else:
            cv.imshow('state', cur_canvas)
            cv.waitKey(0)
            cv.destroyWindow('state')
        # if self.is_finish:
        #     cv.waitKey(0)

    def video(self):
        """TODO: Save the video"""
        pass


def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas