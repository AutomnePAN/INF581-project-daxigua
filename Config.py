# the configuration of the canvas
screen_x = 400
screen_y = 700
end_line = 550


# the configuration of ball types used in the game
# index ordered by the value of radius
balls_setting= {
    0:{'radius':26, 'color':(244,61, 247), 'score': 1, 'name': "grape"},
    1:{'radius':40, 'color':(239,65, 38), 'score': 2, 'name': "cherry"},
    2:{'radius':54, 'color':(111,248, 20), 'score': 3, 'name': "orange"},
    3:{'radius':59.5,'color':(50,205, 249), 'score': 4, 'name': "lemon"},
    4:{'radius':76,'color':(255,240, 44), 'score': 5, 'name': "kiwi"},
    5:{'radius':91.5,'color':(244,61, 247), 'score': 6, 'name': "tomato"},
    6:{'radius':96.5,'color':(251,70, 38), 'score': 7, 'name': "peach"},
    7:{'radius':129,'color':(111, 248, 20), 'score': 8, 'name': "pineapple"},
    8:{'radius':154,'color':(80, 247, 240), 'score': 9, 'name': "coco"},
    9:{'radius':204,'color':(255, 240, 44), 'score': 10, 'name': "watermelon"},
}

# to random a new ball for a game state, we only create balls with level smaller than max_random_level
max_random_ball_level = 3