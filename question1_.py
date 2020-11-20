from Voronoi import Voronoi

import numpy as np
from matplotlib import pyplot as plt

def main():
    x = [43, 20, 34, 18, 12, 32, 40, 4, 44, 30, 6, 47, 23, 13, 38, 48, 36, 46, 50, 37, 21, 7, 28, 25, 10]
    y = [3, 43, 47, 31, 30, 39, 9, 33, 49, 36, 21, 48, 14, 34, 41, 4, 1, 44, 18, 24, 20, 11, 27, 42, 13]
    fig = plt.figure()
    ax=plt.gca()
    points = []
    for i, j in zip(x, y):
        points.append([i, j])
    points = np.asarray(points)
    vp = Voronoi(points)
    vp.process()
    # vp.print_output()
    lines = vp.get_output()

    for line in lines:
        x_list =[line[0], line[2]]
        y_list = [line[1], line[3]]
        ax.plot(x_list, y_list, color='black')
    plt.plot(points[:, 0], points[:, 1], 'bo', markersize=3)
    plt.xlim([0, 53]), plt.ylim([0, 55])
    # plt.savefig('question1_.pdf', dpi=400)
    plt.show()

if __name__ == '__main__':
    main()