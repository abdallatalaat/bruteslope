import math
import numpy as np

HEIGHT = 10
SLOPE = 60
M = 2

DATA = {'slope_angle': SLOPE,
        'h': HEIGHT,
        'radius_range': np.arange(0.5,2.5*M*HEIGHT,0.5),
        'left_right': [M*HEIGHT, 1.5*M*HEIGHT],
        'slope_coordinates': [(0,0), (HEIGHT/math.tan(math.radians(SLOPE)),HEIGHT)],
        'steps_number': 100,
        'below_depth': 0.5*HEIGHT,
        'density': 18,
        'coordinates': [(-2.6*HEIGHT,0), (2*HEIGHT,HEIGHT)],
        'radius': HEIGHT*3,
        }


def update_data(new_height=HEIGHT, new_slope=SLOPE, new_m=M):
    global DATA, SLOPE, M, HEIGHT

    HEIGHT=new_height
    SLOPE=new_slope
    M=new_m

    DATA = {'slope_angle': SLOPE,
            'h': HEIGHT,
            'radius_range': np.arange(0.5, 2.5 * M * HEIGHT, 0.5),
            'left_right': [M * HEIGHT, 1.5 * M * HEIGHT],
            'slope_coordinates': [(0, 0), (HEIGHT / math.tan(math.radians(SLOPE)), HEIGHT)],
            'steps_number': 100,
            'below_depth': 0.5 * HEIGHT,
            'density': 18,
            'coordinates': [(-2.6 * HEIGHT, 0), (2 * HEIGHT, HEIGHT)],
            'radius': HEIGHT * 3,
            }


print(DATA)

update_data(new_slope=30, new_height=15)

print(DATA)