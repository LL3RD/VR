import numpy as np
from numpy.linalg import det
from itertools import combinations

def support(points_list, vector):
    support_point = None
    max_dist = -999
    for point in points_list:
        dist = np.dot(point, vector)
        if support_point is None:
            support_point = point
            max_dist = dist
            continue
        if max_dist < dist:
            support_point = point
            max_dist = dist
    return support_point


def support2(points_list1, points_list2, vector):
    # 变成单位法向量
    if vector.any() != 0:
        vector = vector / np.sqrt(np.sum(vector ** 2))

    p1 = support(points_list1, vector)
    p2 = support(points_list2, -vector)
    # Minkowski
    return np.array(p1 - p2)


def Getlen(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2) + np.power(p1[2] - p2[2], 2))


def GetTriangleArea(p1, p2, p3):
    len1 = Getlen(p1, p2)
    len2 = Getlen(p1, p3)
    len3 = Getlen(p3, p2)

    # Heron's formula
    # https://en.wikipedia.org/wiki/Heron%27s_formula
    p = (len1 + len2 + len3) / 2
    area = np.sqrt(p * (p - len1) * (p - len2) * (p - len3))
    return area


def GetVolume(p1, p2, p3, p4):
        # https://zhuanlan.zhihu.com/p/26140241
    lis = np.asarray([[p1[0], p1[1], p1[2], 1],
                      [p2[0], p2[1], p2[2], 1],
                      [p3[0], p3[1], p3[2], 1],
                      [p4[0], p4[1], p4[2], 1],
                      ])
    volume = det(lis) / 6
    return volume


def Cal_Normal_3D(p1, p2, p3):
    # https://blog.csdn.net/sinat_41104353/article/details/84963016
    # D = p1p2 X p1p3 向量叉乘
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    z1 = p2[2] - p1[2]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    z2 = p3[2] - p1[2]

    a = y1 * z2 - y2 * z1
    b = z1 * x2 - z2 * x1
    c = x1 * y2 - x2 * y1

    normal_vector = np.asarray([a, b, c])

    # 确定法向量是否指向原点
    # 判断向量点乘是否小于0即可
    if np.dot(normal_vector, p1) > 0:
        return -normal_vector
    else:
        return normal_vector


def Cal_Normal_2D(p1, p2):
    # 通过点积求出原点在线段上的投影距离，然后偏移一下即可
    p1O = 0 - p1
    p1p2 = p2 - p1
    dist = np.dot(p1O, p1p2)/np.sqrt(np.sum(p1p2**2))
    # 将距离在p1 p2方向上用向量表示
    dist_vector = p1p2/np.sqrt(np.sum(p1p2**2))*dist
    # 偏移
    vectorD = p1 + dist_vector

    return -vectorD


def NearestSimplex(s):
    origin = np.asarray([0, 0, 0])

    if len(s) == 2:  # 有两个点，为1-simplex，判断线段上是否包含原点
        if Getlen(s[0], s[1]) == Getlen(s[0], origin) + Getlen(s[1], origin):
            return None, None, True
        else:
            vectorD = Cal_Normal_2D(s[0], s[1])
    elif len(s) == 3:  # 有三个点， 为2-simplex,判断三角形中是否包含原点
        if GetTriangleArea(s[0], s[1], s[2]) == GetTriangleArea(s[0], s[1], origin) \
                + GetTriangleArea(s[0], s[2], origin) + GetTriangleArea(s[1], s[2], origin):
            return None, None, True
        else:
            vectorD = Cal_Normal_3D(s[0], s[1], s[2])
    elif len(s) == 4:  # 有四个点， 为3-simplex，判断立方体中是否包含原点
        if GetVolume(s[0], s[1], s[2], s[3]) == GetVolume(s[0], s[1], s[2], origin) + \
                GetVolume(s[0], s[1], origin, s[3]) + GetVolume(s[0], origin, s[2], s[3]) + \
                GetVolume(origin, s[1], s[2], s[3]):
            return None, None, True
        else:
            del s[0]  # 移除掉最早的那个点
            vectorD = Cal_Normal_3D(s[0], s[1], s[2])

    return s, vectorD, False


def GJK_intersection(shape1, shape2, vectorD):
    vectorA = support2(shape1, shape2, vectorD)
    s = [vectorA]
    vectorD = -vectorA
    while True:
        vectorA = support2(shape1, shape2, vectorD)
        if np.dot(vectorA, vectorD) < 0:
            return False
        s.append(vectorA)
        s, vectorD, contains_origin = NearestSimplex(s)
        if contains_origin:
            return True


ObjectA = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
ObjectB = np.asarray([(1, 1, 1), (2, 1, 1), (2, 2, 1), (1, 2, 1), (1, 1, 2), (2, 1, 2), (2, 2, 2), (1, 2, 2)])
ObjectC = np.asarray([(3, 3, 2), (5, 3, 2), (4, 5, 2), (4, 4, 4)])

Objects = {'ObjectA', 'ObjectB', 'ObjectC'}

if __name__ == '__main__':
    for objects in list(combinations(Objects, 2)):
        if GJK_intersection(eval(objects[0]), eval(objects[1]), np.asarray([1, 1, 1])):
            print(objects[0] +' Has a collision with ' + objects[1] + '!!')
        else:
            print(objects[0] +' and ' + objects[1] + ' are not in collision!')

