def poly_area_calculation(points):
    """gets Area and CG"""
    l = len(points)
    if len(points) < 3: raise ValueError('NOT SUFFICIENT POINTS!')

    sum = [0, 0]
    area = 0

    for i in range(l):
        j = i + 1
        if i == l - 1: j = 0

        m = points[i][0]*points[j][1] - points[j][0]*points[i][1]
        print(m)
        sum[0] += (points[i][0]+points[j][0]) * m
        sum[1] += (points[i][1]+points[j][1]) * m
        area += m

    area = 0.5 * area
    sum[0] = sum[0]/(6*area)
    sum[1] = sum[1]/(6*area)

    return [(sum[0], sum[1]), abs(area)]

