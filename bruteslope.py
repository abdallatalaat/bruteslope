# Imports
import math

# Global Variables
DATA = {'slope_angle': 50, 'h': 10} # d_phi: phi in degrees, h: slope vertical height

# Classes

class Cohesive_slope:
    def __init__(self, slope_angle, h, below_depth, density, coordinates, radius):
        self.slope_angle = slope_angle
        self.h = h
        self.density = density
        self.coordinates = coordinates # list of two tuples (two intersection points with the ground)
        self.below_depth = below_depth # depth to the stratum below from the top

        self.slope_coordinates = [(0.0, 0.0),
                                  (h* math.cos(math.radians(slope_angle)), h*math.sin(math.radians(slope_angle)))]
        self.radius = radius

        self.type = self.determine_type()

        self.circle_cg, self.circle_angle = get_circle_centre(self.coordinates, radius)
        self.areas = self.get_areas() # list of lists; each list has [0]: area, [1]: coordinate tuple
        self.cd = self.calculate_cd()
        self.stability_number = self.cd / (self.density * self.h)


    def determine_type(self):
        if self.coordinates[0][0] < 0: return "Base"
        else: return "Slope"

    def get_areas(self):
        if self.circle_cg[0] < self.coordinates[0][0]:
            # No Resisting Areas
            segment = deal_with_arcs(self.coordinates, self.radius)
            poly_coor = [self.coordinates[0], self.slope_coordinates[1], self.coordinates[1]]
            poly = [poly_area_calculation(poly_coor), get_cg(poly_coor)]

            return [segment, poly]
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

                return [segment, poly, res_segment, res_poly]

            else:
                # Resisting areas
                res_segment = deal_with_arcs([self.coordinates[0], int_coor[1]], self.radius)
                res_poly_coor = [self.coordinates[0], (0.0,0.0), int_coor[0], int_coor[1]]
                res_poly = [poly_area_calculation(res_poly_coor), get_cg(res_poly_coor)]

                return [segment, poly, res_segment, res_poly]

    def calculate_cd(self):
        moment = 0
        for area in self.areas:
            moment += area * (area[1][0] - self.circle_cg[0])
        return self.density * moment / (self.circle_angle * self.radius**2)


# Helper Functions


def get_circle_centre(points, radius):
    """returns the cg of a circle (and angle of sector) given two points and a radius"""
    l = distance(points[0], points[1])
    s = line_slope(points)
    phi = 2 * math.atan(l / (2 * radius))
    midpoint = get_cg(points)
    d_circle = math.sqrt(radius ** 2 - 0.25 * l ** 2)
    circle_cg = (midpoint[0] - d_circle * math.sin(s), midpoint[0] + d_circle * math.cos(s))

    return circle_cg, phi

def deal_with_arcs(points, radius):
    """deals with circular segments"""
    l = distance(points[0], points[1])
    phi = 2* math.atan(l/(2*radius))
    s = line_slope(points)
    area = (radius**2/2.0)*(phi-math.sin(phi))
    midpoint = get_cg(points)
    d_circle = math.sqrt(radius**2-0.25*l**2)
    d_segment = (4*radius*(math.sin(phi/2))**3/(3*(phi-math.sin(phi)))) - d_circle
    segment_cg = (midpoint[0]+d_segment*math.sin(s), midpoint[0]-d_segment*math.cos(s))

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
