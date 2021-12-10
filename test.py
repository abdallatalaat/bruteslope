# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Creating vectors X and Y
x = np.linspace(-2, 2, 100)
r = 2
y = -1 * np.sqrt(r**2 - x**2)

fig = plt.figure()
# Create the plot
plt.plot(x, y)

# Show the plot
plt.show()