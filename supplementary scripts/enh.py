#slope enhancer

def slope_enhancer(height, slope_angle, steps_number, coor, ori_radius, below_level=-1, density=18):

    land_lines = [
        [(min(coor[0],0.0)-height, 0.0), (0.0, 0.0)],
        [(0.0, 0.0), (height / math.tan(math.radians(slope_angle)), height)],
        [(height / math.tan(math.radians(slope_angle)), height), (max(coor[1], height / math.tan(math.radians(slope_angle))) + height, height)]
    ]
    vl_lines = [
        [(coor[0], 2*height), (coor[0],-2*height)],
        [(coor[1], 2*height), (coor[1],-2*height)]
    ]

    arc_coordinates = [land_intersect(land_lines, vl_lines[0]), land_intersect(land_lines, vl_lines[1])]

    critical_slope  = Cohesive_slope(slope_angle, height, below_level, density, arc_coordinates, land_lines[1], ori_radius)
    cd_critical = critical_slope.cd

    i_steps = np.linspace(coor[0]-height/2, coor[0]+height/2, steps_number)
    j_steps = np.linspace(coor[1]-height/2, coor[1]+height/2, steps_number)


    if 0 not in i_steps: i_steps = np.append(i_steps, [0])
    if height / math.tan(math.radians(slope_angle)) not in j_steps: j_steps = np.append(j_steps, [height / math.tan(math.radians(slope_angle))])

    radius_range = np.arange(0.75 * ori_radius, 1.5 * ori_radius, 0.1)

    for radius in radius_range:
        print("RADIUS ", radius)
        for i in i_steps:
            for j in j_steps:

                # check if i and j are appropriate, then calculate their y coordinate
                if j <= i or j < 0: continue
                if i <= 0:
                    iy = 0.0
                elif i < height / math.tan(math.radians(slope_angle)):
                    iy = i * math.tan(math.radians(slope_angle))
                else:
                    break
                if j < height / math.tan(math.radians(slope_angle)):
                    jy = j * math.tan(math.radians(slope_angle))
                else:
                    jy = height

                arc_coordinates = [(i, iy), (j, jy)]

                # checks if arc intersection coordinates are appropriate
                if distance(arc_coordinates[0], arc_coordinates[1]) > 2 * radius: continue
                if below_level == 0 and arc_coordinates[0][0] < 0: continue

                iter_slope = Cohesive_slope(slope_angle, height, below_level, density, arc_coordinates, land_lines[1], radius)

                # check to ignore compound circles
                # if iter_slope.compound: continue

                # sanity checks for the iter slope
                if iter_slope.compound:
                    if iter_slope.compound_coor[0][0] < iter_slope.coordinates[0][0]: continue
                if iter_slope.type != "Slope" and distance((0, 0), iter_slope.circle_cg) > radius: continue
                if iter_slope.type == 'Base' and iter_slope.circle_cg[0] < 0: continue
                if (iter_slope.circle_cg[1] - radius > iter_slope.circle_cg[0] * math.tan(
                    math.radians(slope_angle))) and iter_slope.circle_cg[0] > 0: continue

                if iter_slope.cd > cd_critical:
                    critical_slope = iter_slope
                    cd_critical = critical_slope.cd
                    # iter_slope.plot_slope(color='red', width=1, alpha=0.2)
                    # iter_slope.plot_circle_cg(alpha=0.4)

    return critical_slope

output_file = 'enhanced2.csv'
out = open(output_file, 'a')
file_name = "stewardstep400radius10th.csv"
data = parse_slope_data(file_name)
for a in data:
    print('LEVEL ' + str(a[0]) + ' ANGLE ' + str(a[1]))

    out.close()
    out = open(output_file, 'a')

    coor = reverse_slope_coor(a[3], a[4], 10, a[1], a[-1])
    mySlope = slope_enhancer(HEIGHT, a[1], 100, coor, a[-2], below_level=(a[0]-1)*HEIGHT, density=18)

    # below depth, angle, stability number n, radius, circle x, circle y, slope type, compound?, intersection coor x, intersection coor y
    iter_data = [a[0], a[1], mySlope.stability_number, mySlope.radius, mySlope.circle_cg[0], mySlope.circle_cg[1],
                 mySlope.type, mySlope.compound, mySlope.coordinates[0][0], mySlope.coordinates[1][0]]



    out.write(write_list(iter_data))
    print(mySlope)

out.close()