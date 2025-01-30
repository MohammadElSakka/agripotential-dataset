import matplotlib.pyplot as plt
image_path = "../data/dataset/binary_mask.png" 
img = plt.imread(image_path)
up, down, left, right =  3321, 10979, 0, 9401
image = img[up:down+1, left:right+1]
fig, ax = plt.subplots()
ax.imshow(image)
vertices = []
def onclick(event):
    if event.button == 1:  
        vertices.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro')
        if len(vertices) > 1:
            x, y = zip(*vertices)
            ax.plot(x, y, 'b-')
        fig.canvas.draw()
def onkey(event):
    if event.key == 'enter': 
        if len(vertices) > 2:
            print("Polygon vertices:", vertices)
cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
plt.show()