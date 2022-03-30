import imageio, os

images = []
for file_name in [str(i)+'.png' for i in range(89*4,89*5)]:
    file_path = os.path.join("anim_loop", file_name)
    images.append(imageio.imread(file_path))

images2 = list(images)
images2.reverse()
for i in images2: images.append(i)
imageio.mimsave('animation5.gif', images, fps=30)