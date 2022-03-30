import math

HEIGHT = 10

def reverse_slope_coor(circle_cg, radius, height, slope_angle, stype):
    """returns (i,j): x_coor for land intersection"""
    point_left = [0,0]
    point_right = [0,0]
    if stype == " Base":
        point_left[1]=0
        point_right[1]=height
        point_left[0] = circle_cg[0] - math.sqrt(radius**2-circle_cg[1]**2)
        point_right[0] = circle_cg[0] + math.sqrt(radius**2-(height-circle_cg[1])**2)

    elif stype == " Toe":
        point_right = [circle_cg[0] + math.sqrt(radius**2-(height-circle_cg[1])**2), height]

    elif stype == " Slope":
        horizontal_left = [circle_cg[0] - math.sqrt(radius**2-circle_cg[1]**2), 0]
        horizontal_right = [circle_cg[0] + math.sqrt(radius**2-(height-circle_cg[1])**2), height]

        alpha = math.tan(math.radians(slope_angle))

        a = 1+alpha**2
        b = -2*(circle_cg[0]+circle_cg[1]*alpha)
        c = circle_cg[0]**2+circle_cg[1]**2-radius**2

        slope_left_x = (1/2*a)*(-1*b-math.sqrt(b**2-4*a*c))
        slope_right_x = (1/2*a)*(-1*b+math.sqrt(b**2-4*a*c))

        slope_left = [slope_left_x, slope_left_x*alpha]
        slope_right = [slope_right_x, slope_right_x*alpha]


        if slope_left[1] < 0: point_left = horizontal_left
        else: point_left = slope_left

        if slope_right[1] > height: point_right = horizontal_right
        else: point_right = slope_right


    return [point_left, point_right]

def parse_slope_data(file_name):
    global HEIGHT
    """returns a list of slope data present in a csv:
    depth, slope, m, radius, circle_x, circle_y, ...
    each entry has [below_depth, angle, m, circle_coor, radius, stype]"""


    f = open(file_name, 'r')
    output = []

    for line in f:
        lid = line.split(",")
        #if lid[0] == 'depth': continue

        below_depth = float(lid[0])
        angle = int(lid[1])
        m = float(lid[2])
        radius = float(lid[3])
        circle_coor = (float(lid[4]), float(lid[5]))
        stype = lid[6]
        intersection_coor = reverse_slope_coor(circle_coor, radius, HEIGHT, angle, stype)


        output.append([below_depth, angle, m, circle_coor, radius, stype, intersection_coor])

    f.close()

    return output


#########################################
file_name = "FINAL SLOPE DATA.csv"

data = parse_slope_data(file_name)
