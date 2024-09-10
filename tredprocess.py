import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def show_img(img, label):
    x, y = [], []
    for i in range(1, 9):
        if i % 2 == 1:
            x.append(label[i-1])
        else:
            y.append(label[i-1])
    for i in range(0, len(x), 2):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-')
    plt.plot([x[0], x[3]], [y[0], y[3]], 'ro-')
    plt.plot([x[1], x[2]], [y[1], y[2]], 'ro-')
    plt.imshow(img)
    plt.show()
