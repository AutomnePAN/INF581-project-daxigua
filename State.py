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

    def vectorize(self, grid_size = 40 ):
        """
        vectorize current state by a vector of dimension 3*N + 1
        We suppose there is only one ball above the endline
        """
        M = int(self.endline / grid_size) + 1
        N = int(self.screen_x / grid_size) + 1
        
        result = np.zeros(M * N + 1)
        self.balls.sort(key = lambda b: b.position[1], reverse=True)
        k = 0
        for i in range( len(self.balls)):
            if self.balls[i].position[1] > self.endline:
                result[0] = 2 * self.balls[i].radius / self.screen_x
            else:
                grid_x = int(self.balls[i].position[0] / grid_size)
                grid_y = int(self.balls[i].position[1] / grid_size)
                result[1 + N * grid_y + grid_x ] = 2 * self.balls[i].radius / self.screen_x
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
        """Show the game as a video"""
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
                      (int(ball.position[0]), int(self.screen_y - ball.position[1])),
                      int(ball.radius), ball.color, -1)

            text = f"{ball.ball_level + 1}"
            textsize = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv.putText(cur_canvas,
                       text,
                       (int(ball.position[0] - textsize[0] / 2),
                        int(self.screen_y - ball.position[1] + textsize[1] / 2)),
                       cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv.putText(cur_canvas, "Score : {}".format(self.score), (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255),
                   2)
        cv.imshow('video', cur_canvas)
        cv.waitKey(1)


def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas