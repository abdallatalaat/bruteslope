from cleaner_bruteslope import *

file_name = "FINAL SLOPE DATA.csv"

data = parse_slope_data(file_name)

for r in range(len(data)):

    a = data[r]

    fig,ax = plt.subplots(figsize=(12,10))

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_ylim(-a[0]*HEIGHT, 30)
    ax.set_xlim(-a[0]*HEIGHT, a[0]*HEIGHT+20)

    slope_coor = [[0,0], [HEIGHT/math.tan(math.radians(a[1])),HEIGHT]]
    iter_slope = Cohesive_slope(a[1], HEIGHT, (a[0]-1)*HEIGHT, DATA['density'],a[-1],slope_coor,a[4])

    plot_land(iter_slope, True)
    iter_slope.plot_slope(color='blue', width=3, style='solid')
    #iter_slope.plot_circle_cg()

    ax.text(-10, 22, "Slope Angle: {:}".format(a[1]), ha='left', color='red', weight='black')
    ax.text(-10, 20, "D/H: {:.1f}".format(a[0]), ha='left', color='red', weight='black')

    plt.savefig("anim_loop/" + str(r) + ".png")
    print(a[0],a[1])

    plt.close()