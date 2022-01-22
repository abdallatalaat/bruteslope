def land_intersect(land_lines, line):
    """returns intersection point with predefined land"""

    for land_line in land_lines:
        l = intersection_point(land_line, line)
        if l != None: return l

    return None
