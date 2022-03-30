Ds = [1, 1.2, 1.5, 2, 4]

file_name = "f.csv"
f = open(file_name, 'r')

for depth in Ds:
    print('\n\n DEALING WITH ALL D = ', depth-1)


    DATA['below_depth'] = (depth - 1) * HEIGHT

    for angle in range(2,90,1):

        if angle %5 ==0: continue
        if angle >= 50 and angle <=60: continue


        f.close()
        f = open(file_name, 'a')


        update_data(new_slope=angle, radius_step=0.1, steps_number=400)
        DATA['below_depth'] = (depth - 1) * HEIGHT



        mySlope = generate_failures(DATA['h'],
                                    DATA['slope_angle'],
                                    DATA['steps_number'],
                                    DATA['left_right'],
                                    DATA['radius_range'],
                                    below_level=DATA['below_depth'],
                                    density=DATA['density'], plot=False)

        iter_data = [depth, SLOPE, mySlope.stability_number, mySlope.radius, mySlope.circle_cg[0], mySlope.circle_cg[1], mySlope.type, mySlope.compound]

        f.write(write_list(iter_data))

f.close()

































for angle in range(10,90,5):
    plt.clf()

    update_data(new_slope=angle)

    mySlope = generate_failures(DATA['h'],
                                DATA['slope_angle'],
                                DATA['steps_number'],
                                DATA['left_right'],
                                DATA['radius_range'],
                                below_level=DATA['below_depth'],
                                density=DATA['density'])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-10,70)
    plt.ylim(-1*DATA['below_depth']-1,50)

    plt.savefig('plots/' + str(angle) + '.png')
