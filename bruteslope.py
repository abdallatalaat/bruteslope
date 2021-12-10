# Imports
import math
import numpy as np
import matplotlib.pyplot as plt

# Global Variables
DATA = {'slope_angle': 60, 'h': 10, 'coordinates': [(0,0), (15,10)]}

# Classes

class Cohesive_slope:
    def __init__(self, slope_angle, h, below_depth, density, coordinates, slope_coor, radius):
        self.slope_angle = slope_angle
        self.h = h
        self.density = density
        self.coordinates = coordinates # list of two tuples (two intersection points with the ground)
        self.below_depth = below_depth # depth to the stratum below from the top
        self.slope_coordinates = slope_coor
        self.radius = radius

        self.type = self.determine_type()

        self.circle_cg, self.circle_angle = get_circle_centre(self.coordinates, radius)
        self.areas, self.ne_poly, self.po_poly = self.get_areas() # list of lists; each list has [0]: area, [1]: coordinate tuple

        self.cd = self.calculate_cd()
        self.stability_number = self.cd / (self.density * self.h)


    def __str__(self):
        msg = "\nType: " + self.type
        msg = msg + "\ncircle cg: " + str(self.circle_cg)
        msg = msg + "\ncircle radius: " + str(self.radius)
        msg = msg + "\ncircle angle: " + str(self.circle_angle*180/np.pi)
        msg = msg + "\ndeveloped cohesion (cd): " + str(self.cd)
        msg = msg + "\nstability number (m): " + str(self.stability_number)

        return msg


    def determine_type(self):
        if self.coordinates[0][0] < 0: return "Base"
        else: return "Slope"

    def get_areas(self):

        areas = []
        ne_poly = []
        po_poly = []


        if self.circle_cg[0] < self.coordinates[0][0]:
            # No Resisting Areas
            segment = deal_with_arcs(self.coordinates, self.radius)
            poly_coor = [self.coordinates[0], self.slope_coordinates[1], self.coordinates[1]]
            poly = [poly_area_calculation(poly_coor), get_cg(poly_coor)]

            self.poly_coordinates = poly_coor
            areas = [segment, poly]
            po_poly = poly_coor


        else:
            int_coor = [intersection_point([self.circle_cg, [self.circle_cg[0], -100*self.radius]], self.slope_coordinates),
                        [self.circle_cg[0], self.circle_cg[1]-self.radius]]


            # Acting Areas
            segment= deal_with_arcs([int_coor[1], self.coordinates[1]], self.radius)
            poly_coor = [int_coor[0], int_coor[1], self.coordinates[1], self.slope_coordinates[1]]

            poly = [poly_area_calculation(poly_coor), get_cg(poly_coor)]

            if self.type == "Slope":
                # Resisting areas
                res_segment = deal_with_arcs([self.coordinates[0], int_coor[1]], self.radius)
                res_poly_coor = [self.coordinates[0], int_coor[0], int_coor[1]]
                res_poly = [poly_area_calculation(res_poly_coor), get_cg(res_poly_coor)]

                areas = [segment, poly, res_segment, res_poly]

            else:
                # Resisting areas
                res_segment = deal_with_arcs([self.coordinates[0], int_coor[1]], self.radius)
                res_poly_coor = [self.coordinates[0], (0.0,0.0), int_coor[0], int_coor[1]]
                res_poly = [poly_area_calculation(res_poly_coor), get_cg(res_poly_coor)]

                areas = [segment, poly, res_segment, res_poly]

            ne_poly = res_poly_coor+[res_poly_coor[0]]
            po_poly = poly_coor + [poly_coor[0]]


        return areas, ne_poly, po_poly


    def calculate_cd(self):
        moment = 0
        for area in self.areas:
            moment += area[0] * (area[1][0] - self.circle_cg[0])
        return (self.density * moment) / (self.circle_angle * self.radius**2)

    def plot_slope(self, color="grey", width=0.5, style="dashed"):
        plot_arc(self.coordinates,self.circle_cg,self.radius,color,width,style)

    def plot_poly(self):
        #shapes

        x = []
        y = []
        for cor in self.ne_poly:
            x.append(cor[0])
            y.append(cor[1])
        plt.plot(x,y)

        x = []
        y = []
        for cor in self.po_poly:
            x.append(cor[0])
            y.append(cor[1])
        plt.plot(x,y)


        #CGS
        x = []
        y = []
        for area in self.areas:
            x.append(area[1][0])
            y.append(area[1][1])

            plt.text(area[1][0], area[1][1],"{:.2f}, ({:.2f})".format(area[0], area[1][0]-self.circle_cg[0]))

        plt.scatter(x,y)


# Helper Functions

def plot_arc(points, cg, radius, color="grey", width=0.5, style="dashed"):
    theta_1 = line_slope([points[0], cg]) + np.pi
    theta_2 = line_slope([cg, points[1]]) + 2 * np.pi

    if theta_1 < np.pi: theta_1 += np.pi

    theta = np.linspace(theta_1, theta_2, 100)

    x = radius * np.cos(theta) + cg[0]
    y = radius * np.sin(theta) + cg[1]
    plt.plot(x, y, color, linewidth=width, linestyle=style)



def get_circle_centre(points, radius):
    """returns the cg of a circle (and angle of sector) given two points and a radius"""
    l = distance(points[0], points[1])
    s = line_slope(points)
    phi = math.pi - 2 * math.acos(l / (2 * radius))
    midpoint = (0.5*(points[0][0]+points[1][0]), 0.5*(points[0][1]+points[1][1]))
    d_circle = math.sqrt(radius ** 2 - (0.5 * l) ** 2)
    circle_cg = (midpoint[0] - d_circle * math.sin(s), midpoint[1] + d_circle * math.cos(s))

    return circle_cg, phi

def deal_with_arcs(points, radius):
    """deals with circular segments"""
    l = distance(points[0], points[1])
    phi = math.pi - 2 * math.acos(l / (2 * radius))
    s = line_slope(points)
    area = (radius**2/2.0)*(phi-math.sin(phi))
    midpoint = get_cg(points)
    d_circle = math.sqrt(radius**2-0.25*l**2)
    d_segment = (4*radius*(math.sin(phi/2))**3/(3*(phi-math.sin(phi)))) - d_circle
    segment_cg = (midpoint[0]+d_segment*math.sin(s), midpoint[1]-d_segment*math.cos(s))

    return [area, segment_cg]

def poly_area_calculation(list_of_coordinates):
    """takes a list of 2D-coordinates(tuples) and reutrns area"""
    if len(list_of_coordinates) == 0: return 0
    area = 0
    l = len(list_of_coordinates)
    for iter in range(l-1):
        area = area + list_of_coordinates[iter][0]*list_of_coordinates[iter+1][1] - list_of_coordinates[iter][1]*list_of_coordinates[iter+1][0]
    area = area + list_of_coordinates[l-1][0]*list_of_coordinates[0][1] - list_of_coordinates[l-1][1]*list_of_coordinates[0][0]
    return abs(area/2)
def get_cg(list_of_coordinates):
    """gets CG of list of coordinates"""
    l = len(list_of_coordinates)
    if l == 0 : return [0, 0]
    s = [0.0, 0.0]
    for coor in list_of_coordinates:
        s[0] += coor[0]
        s[1] += coor[1]

    return(s[0]/l, s[1]/l)
def distance(point1, point2):
    """returns distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
def isBetween(middle_point, line):
    """checks if a point lies on a st. line between two other points"""
    if distance(line[0], middle_point) <= distance(line[0], line[1]): return True
    return False
def intersection_point(line1, line2):
    """takes lines as a list of two 2D coordinates(tuples) and returns the coordinate of their point of intersection.
    if no intersection is found returns None"""

    # slopes

    if (line1[0][0] - line1[1][0]) != 0:
        m1 = (line1[0][1] - line1[1][1])/(line1[0][0] - line1[1][0])
        param1 = [m1, -1, line1[0][1] - line1[0][0] * m1]
    else:
        m1 = float('inf')
        param1 = [1, 0, -1 * line1[0][0]]

    if (line2[0][0] - line2[1][0]) != 0:
        m2 = (line2[0][1] - line2[1][1])/(line2[0][0] - line2[1][0])
        param2 = [m2, -1, line2[0][1] - line2[0][0]*m2]
    else:
        m2 = float('inf')
        param2 = [1, 0, -1 * line2[0][0]]

    # return None if parallel
    if m1 == m2: return None

    solution =  ((param1[1]*param2[2] - param2[1]*param1[2])/(param1[0]*param2[1] - param2[0]*param1[1]),
            (param1[2]*param2[0] - param2[2]*param1[0])/(param1[0]*param2[1] - param2[0]*param1[1]))

    # checks if intersection point lies inside line segment
    if not isBetween(solution, line1): return None

    return solution
def line_slope(line):
    """
    :param line: list of two points
    :return: slope of line
    """
    return math.atan((line[1][1]-line[0][1])/(line[1][0]-line[0][0]))

def min_radius(h, angle):
    return h/math.sin(math.radians(angle))

# brains
def generate_failures(height, slope_angle, steps_number, half_horiz, radius_range):
    """
    generates multiple Cohesive_slope objects
    :param height: vertical height
    :param slope_angle: soil angle IN DEGREES
    :param steps_number: number of steps
    :param half_horiz: horizontal surveyed distance before and after the slope.
    :param radius_range: linespace of radii
    :return: list of Cohesive_slope objects, the critical slope, and the slope coordinates
    """
    land_coor = [(-1.0*half_horiz, 0.0),
                 (0.0, 0.0),
                 (height / math.tan(math.radians(slope_angle)), height),
                 ((height / math.tan(math.radians(slope_angle)))+half_horiz, height)]

    horizontal_step = (2*half_horiz + (height / math.tan(math.radians(slope_angle))))/steps_number

    cd_critical = float('-inf')
    slopes = []
    critical_slope = None

    horiz_steps = np.linspace(-1.0*half_horiz, (height / math.tan(math.radians(slope_angle)))+half_horiz, steps_number)
    if 0 not in horiz_steps: horiz_steps = np.append(horiz_steps, [0])
    if land_coor[2][0] not in horiz_steps: horiz_steps = np.append(horiz_steps, [land_coor[2][0]])

    print(horiz_steps)

    for radius in radius_range:
        print(" Computing all possible circles of radius {:.5f}".format(radius))
        for i in horiz_steps:
            for j in horiz_steps:
                if j <= i or j < 0: continue

                if i <=0: iy = 0.0
                elif i<land_coor[2][0]: iy = i * math.tan(math.radians(slope_angle))
                else: break

                if j < land_coor[2][0]: jy = j * math.tan(math.radians(slope_angle))
                else: jy = height

                arc_coordinates = [(i, iy), (j, jy)]
                if distance(arc_coordinates[0], arc_coordinates[1]) > 2*radius: continue

                iter_slope = Cohesive_slope(slope_angle,height,0,18,arc_coordinates,slope_coor,radius)

                if iter_slope.type=="Base" and distance((0, 0), iter_slope.circle_cg) > radius: continue
                if iter_slope.circle_cg[0] < 0: continue

                slopes.append(iter_slope)
                if iter_slope.cd > cd_critical:
                    critical_slope = iter_slope
                    cd_critical = critical_slope.cd

    return slopes, critical_slope, slope_coor








fig = plt.figure()

slope_coor = [(0.0, 0.0), (DATA['h'] / math.tan(math.radians(DATA['slope_angle'])), DATA['h'])]


r_step = 0.001
r_min = 14
r_max = 15
half_horiz = 15


radius_range = np.arange(r_min, r_max, r_step)

slopes, critical_slope, slope_coor = generate_failures(DATA['h'], DATA['slope_angle'],200,half_horiz, radius_range)


print("\ndrawing slope..")


critical_slope.plot_slope("blue", 1.5, "solid")
print("\n\n")
print(critical_slope)


#
# mySlope = Cohesive_slope(60,10,0,18,[(0,0),(15,10)], slope_coor,10)
# mySlope.plot_slope(color="blue")


plt.plot([slope_coor[0][0] - half_horiz-2, slope_coor[0][0], slope_coor[1][0], slope_coor[1][0] + half_horiz+2],
         [slope_coor[0][1], slope_coor[0][1], slope_coor[1][1], slope_coor[1][1]],
         "black", linewidth=3)


plt.gca().set_aspect('equal', adjustable='box')
plt.show()

