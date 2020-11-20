import numpy as np
from numpy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

ObjectA = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
ObjectB = np.asarray([(1, 1, 1), (2, 1, 1), (2, 2, 1), (1, 2, 1), (1, 1, 2), (2, 1, 2), (2, 2, 2), (1, 2, 2)])
ObjectC = np.asarray([(3, 3, 2), (5, 3, 2), (4, 5, 2), (4, 4, 4)])

Objects = {'ObjectA': 'Box1', 'ObjectB': 'Box2', 'ObjectC': 'Box3'}


def Getlen(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2) + np.power(p1[2] - p2[2], 2))


def AABB_Box(points):
    MAX = np.max(points, axis=0)
    MIN = np.min(points, axis=0)
    p1 = np.asarray([MAX[0], MAX[1], MAX[2]])
    p2 = np.asarray([MAX[0], MAX[1], MIN[2]])
    p3 = np.asarray([MAX[0], MIN[1], MIN[2]])
    p4 = np.asarray([MAX[0], MIN[1], MAX[2]])
    p5 = np.asarray([MIN[0], MIN[1], MAX[2]])
    p6 = np.asarray([MIN[0], MAX[1], MAX[2]])
    p7 = np.asarray([MIN[0], MAX[1], MIN[2]])
    p8 = np.asarray([MIN[0], MIN[1], MIN[2]])

    Box = np.asarray([p1, p2, p3, p4, p5, p6, p7, p8])
    L = np.sum(p1 - p2)
    W = np.sum(p1 - p4)
    H = np.sum(p1 - p6)
    return Box, L, W, H


def OBB_Box(points):
    # 协方差矩阵
    covariance_matrix = np.cov(points, y=None, rowvar=0, bias=1)
    # 特征向量
    _, eigen_vectors = np.linalg.eigh(covariance_matrix)

    def try_to_normalize(v):
        n = np.linalg.norm(v)
        if n < np.finfo(float).resolution:
            raise ZeroDivisionError
        return v / n

    def transform(point, rotation):
        return np.dot(np.array(point), rotation)

    r = try_to_normalize(eigen_vectors[:, 0])
    u = try_to_normalize(eigen_vectors[:, 1])
    f = try_to_normalize(eigen_vectors[:, 2])

    # OBB的旋转角度
    rotation = np.asarray((r, u, f)).T

    p_primes = np.asarray([rotation.dot(p) for p in points])

    # OBB上的最小/最大点
    MIN = np.min(p_primes, axis=0)
    MAX = np.max(p_primes, axis=0)

    # OBB 的中心点
    centroid = transform((MIN + MAX) / 2.0, rotation)

    RTF = transform((MAX[0], MAX[1], MIN[2]), rotation)
    RTC = transform(MAX, rotation)
    LTF = transform((MIN[0], MAX[1], MIN[2]), rotation)
    RBF = transform((MAX[0], MIN[1], MIN[2]), rotation)
    L = Getlen(RTF, RTC)
    H = Getlen(RTF, RBF)
    W = Getlen(RTF, LTF)

    # 8个点
    return np.asarray([
        # upper cap: ccw order in a right-hand system
        # rightmost, topmost, farthest
        RTF,
        # leftmost, topmost, farthest
        LTF,
        # leftmost, topmost, closest
        transform((MIN[0], MAX[1], MAX[2]), rotation),
        # rightmost, topmost, closest
        RTC,
        # lower cap: cw order in a right-hand system
        # leftmost, bottommost, farthest
        transform(MIN, rotation),
        # rightmost, bottommost, farthest
        RBF,
        # rightmost, bottommost, closest
        transform((MAX[0], MIN[1], MAX[2]), rotation),
        # leftmost, bottommost, closest
        transform((MIN[0], MIN[1], MAX[2]), rotation),
    ]), L, W, H


def plot(shape, ax, L=1, W=1, H=1, Triangle=False, color="b", alpha=0.5, linestyle='--'):
    for s, e in combinations(shape, 2):
        Len = Getlen(s, e)
        if Triangle:
            ax.plot3D(*zip(s, e), color=color, alpha=alpha, linestyle=linestyle)
            continue
        if abs(Len - H <= 0.0001) or abs(Len - W <= 0.0001) or abs(Len - L <= 0.0001):
            ax.plot3D(*zip(s, e), color=color, alpha=alpha, linestyle=linestyle)


def collide_check(MAX1, MIN2, MIN1, MAX2):
    return (MIN1 - MAX2) * (MAX1 - MIN2) <= 0


def unit_vector(vector):  # 单位向量
    if vector.any() != 0:
        vector = vector / np.sqrt(np.sum(vector ** 2))
    return vector


def Cal_Normal_3D(v1,v2): # 法向量
    # https://blog.csdn.net/sinat_41104353/article/details/84963016
    # D = v1 X v2 向量叉乘

    a = v1[1] * v2[2] - v2[2] * v1[2]
    b = v1[2] * v2[0] - v2[2] * v1[0]
    c = v1[0] * v2[1] - v2[0] * v1[1]

    normal_vector = np.asarray([a, b, c])
    return unit_vector(normal_vector)


def Collision_Detection_OBB_Box(Box1, Box2):
    # https://www.jkh.me/files/tutorials/Separating%20Axis%20Theorem%20for%20Oriented%20Bounding%20Boxes.pdf
    RTF1, RTF2 = Box1[0], Box2[0]
    RTC1, RTC2 = Box1[3], Box2[3]
    LTF1, LTF2 = Box1[1], Box2[1]
    RBF1, RBF2 = Box1[5], Box2[5]
    LBC1, LBC2 = Box1[7], Box2[7]

    cent1, cent2 = (RTF1 + LBC1) / 2, (RTF2 + LBC2) / 2

    T = cent2 - cent1  # 两盒子中心点向量
    D1, D2 = Getlen(RTF1, RTC1) / 2, Getlen(RTF2, RTC2) / 2  # half depth of box (corresponds with the local z-axis of box)
    H1, H2 = Getlen(RTF1, RBF1) / 2, Getlen(RTF2, RBF2) / 2  # half height of A (corresponds with the local y-axis of A)
    W1, W2 = Getlen(RTF1, LTF1) / 2, Getlen(RTF2, LTF2) / 2  # half width of A (corresponds with the local x-axis of A)

    Ax, Bx = unit_vector(LTF1 - RTF1), unit_vector(LTF2 - RTF2)  # unit vector representing the x-axis
    Ay, By = unit_vector(RBF1 - RTF1), unit_vector(RBF2 - RTF2)  # unit vector representing the y-axis
    Az, Bz = unit_vector(RTC1 - RTF1), unit_vector(RTC2 - RTF2)  # unit vector representing the z-axis

    # |T·L|>|W1x·L|+|H1Ay·L|+|D1Az·L|+|W2Bx·L|+|H1By·L|+|D1Bz·L|
    # CASE 1: L = Ax ...
    CASES = [Ax, Ay, Az, Bx, By, Bz,
             Cal_Normal_3D(Ax, Bx), Cal_Normal_3D(Ax, By), Cal_Normal_3D(Ax, Bz),
             Cal_Normal_3D(Ay, Bx), Cal_Normal_3D(Ay, By), Cal_Normal_3D(Ay, Bz),
             Cal_Normal_3D(Az, Bx), Cal_Normal_3D(Az, By), Cal_Normal_3D(Az, Bz)]

    for L in CASES:
        if np.abs(np.dot(T, L)) > np.abs(np.dot(W1*Ax, L)) + np.abs(np.dot(H1*Ay, L))\
           + np.abs(np.dot(D1*Az, L)) + np.abs(np.dot(W2*Bx, L)) + np.abs(np.dot(H2*By,L))\
           + np.abs(np.dot(D2*Bz, L)):
           return False
    return True

def Collision_Detection_AABB_Box(Box1, Box2):
    x_min = np.min(Box1[:, 0])
    x_max = np.max(Box1[:, 0])
    y_min = np.min(Box1[:, 1])
    y_max = np.max(Box1[:, 1])
    z_min = np.min(Box1[:, 2])
    z_max = np.max(Box1[:, 2])

    xref_min = np.min(Box2[:, 0])
    xref_max = np.max(Box2[:, 0])
    yref_min = np.min(Box2[:, 1])
    yref_max = np.max(Box2[:, 1])
    zref_min = np.min(Box2[:, 2])
    zref_max = np.max(Box2[:, 2])

    x_collide = collide_check(x_max, xref_min, x_min, xref_max)
    y_collide = collide_check(y_max, yref_min, y_min, yref_max)
    z_collide = collide_check(z_max, zref_min, z_min, zref_max)

    return (x_collide and y_collide and z_collide)


def main(ax, method):
    Box1, L1, W1, H1 = eval(method)(ObjectA)
    Box2, L2, W2, H2 = eval(method)(ObjectB)
    Box3, L3, W3, H3 = eval(method)(ObjectC)


    plot(Box3, ax, L3, W3, H3, color='r', alpha=1, linestyle='-')
    plot(Box2, ax, L2, W2, H2, color='r', alpha=1, linestyle=':')
    plot(Box1, ax, L1, W1, H1, color='r', alpha=1, linestyle=':')

    for objects in list(combinations(Objects, 2)):
        if eval('Collision_Detection_{}'.format(method))(eval(Objects[objects[0]]), eval(Objects[objects[1]])):
            print(objects[0] + ' Has a collision with ' + objects[1] + '!!')
        else:
            print(objects[0] + ' and ' + objects[1] + ' are not in collision!')


if __name__ == '__main__':

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 4])
    # ax = fig.gca(projection='3d')
    ax.set_aspect("auto")

    plot(ObjectA, ax, 1)
    plot(ObjectB, ax, 1)
    plot(ObjectC, ax, Triangle=True)

    # AABB_Box // OBB_Box
    main(ax, 'OBB_Box')

    ObjectC_4P = [ObjectC[1:], ObjectC[:4], ObjectC[[0, 2, 3]], ObjectC[[0, 1, 3]]]
    for Ob in ObjectC_4P:
        tri = Poly3DCollection(Ob, alpha=0.5, facecolor='g')
        tri.set_edgecolor('b')
        ax.add_collection3d(tri)
    plt.savefig('OBB.pdf', dpi=400)
    plt.show()
