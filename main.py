from cleaner_bruteslope import *


mySlope = Cohesive_slope(40,10,10,18,[[-15.0342,0],[32.14108,10]],[[0,0],[10/math.tan(math.radians(40)),10]],27.1)

print(mySlope)

fig, ax = plt.subplots()
mySlope.plot_slope(color='blue',width=3,style='solid')
plot_land(mySlope, True)

mySlope.plot_compound(color='blue',width=3,style='solid')
mySlope.plot_poly()
mySlope.plot_circle_cg()
ax.set_xlim(-20,50)
ax.set_ylim(-15,30)
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
plt.show()