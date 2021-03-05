# the configuration of the canvas
screen_x = 252
screen_y = 469
end_line = 400


# the configuration of ball types used in the game
# index ordered by the value of radius
balls_setting= {
    0:{'radius':16, 'color':(244,61, 247), 'score': 1, 'name': "grape"},
    1:{'radius':20, 'color':(239,65, 38), 'score': 2, 'name': "cherry"},
    2:{'radius':34, 'color':(111,248, 20), 'score': 3, 'name': "orange"},
    3:{'radius':39.5,'color':(50,205, 249), 'score': 4, 'name': "lemon"},
    4:{'radius':46,'color':(255,240, 44), 'score': 5, 'name': "kiwi"},
    5:{'radius':51.5,'color':(244,61, 247), 'score': 6, 'name': "tomato"},
    6:{'radius':56.5,'color':(251,70, 38), 'score': 7, 'name': "peach"},
    7:{'radius':69,'color':(111, 248, 20), 'score': 8, 'name': "pineapple"},
    8:{'radius':84,'color':(80, 247, 240), 'score': 9, 'name': "coco"},
    9:{'radius':104,'color':(255, 240, 44), 'score': 10, 'name': "watermelon"},
}

# to random a new ball for a game state, we only create balls with level smaller than max_random_level
max_random_ball_level = 3