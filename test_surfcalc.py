import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           binary_opening)

a = np.array([[1, 0, 0, 1, 1, 0, 0],
              [1, 1, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 1, 0],
              [1, 1, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0]])

a = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [1, 1, 0, 0, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 0, 0, 1, 0],
              ])


def enclosed_by_four(a):
    a2 = a+np.roll(a, 1, axis=0)
    a3 = a2+np.roll(a2, 1, axis=1)
    a3[:, 0] = 0
    a3[0, :] = 0
    return np.array(a3 == 4).astype(int)


def atleast_one(a):
    a2 = a+np.roll(a, 1, axis=0)
    a3 = a2+np.roll(a2, 1, axis=1)
    return np.array(a3 >= 1).astype(int)


a_o = binary_opening(a, structure=[[1, 1], [1, 1]]).astype(int)
a_f = enclosed_by_four(a_o)
a_f_f = enclosed_by_four(a)
a_f2 = binary_erosion(a, structure=np.ones((2, 2)), origin=(0, 0))

print(f"a:\n {a}\n\n a_o:\n {a_o}\n\n a_f:\n {a_f}\n\n a_f_f:\n{a_f_f}\n\n a_f2:\n{a_f2.astype(int)}\n\n")
print(np.sum(np.abs(a_f_f-a_f2)))

ad = np.zeros(np.array(a.shape)+[2, 1])
ad[1:-1, 0:-1] = a
ad_d = binary_dilation(
    ad, structure=np.ones((3, 3)), iterations=1).astype(int)
ad_f = enclosed_by_four(ad_d) & atleast_one(ad)
ad_f_f = enclosed_by_four(binary_dilation(ad, structure=np.ones((3, 3)),
                                          iterations=1).astype(int)) & atleast_one(ad)
ad_f2 = binary_erosion((binary_dilation(ad, structure=np.ones((3, 3)),
                                        iterations=1).astype(int)), structure=np.ones((2, 2)), origin=(0, 0)) & binary_dilation(ad, structure=np.ones((2, 2)), origin=(-1, -1))
print(
    f"ad: \n {ad}\n\n a_o: \n {ad_d}\n\n a_f: \n {ad_f}\n\n{ad_f_f}\n\n ad_f2:\n{ad_f2.astype(int)}")
print(np.sum(np.abs(ad_f2-ad_f_f)))


# Origin works differently for erosion as for dilations see below
#  0  0: [[1, 1], [1, (1)]]
#  0 -1: [[1, 1], [(1), 1]]
# -1  0: [[1, (1)], [1, 1]]
# -1 -1: [[(1), 1], [1, 1]]
ad_ao = atleast_one(ad)
struct = np.ones((2, 2))

ad_d00 = binary_erosion(ad, structure=struct,
                        origin=(0, 0)).astype(int)
ad_d01 = binary_erosion(ad, structure=struct,
                        origin=(0, -1)).astype(int)
ad_d10 = binary_erosion(ad, structure=struct,
                        origin=(-1, 0)).astype(int)
ad_d11 = binary_erosion(ad, structure=struct,
                        origin=(-1, -1)).astype(int)

print(f"ad:\n{ad}\\n\nad_d00:\n{ad_d00}\n\nad_d01:\n{ad_d01}\n\nad_d10:\n{ad_d10}\n\nad_d11:\n{ad_d11}")

# dilation
#  0  0: [[(1), 1], [1, 1]]
#  0 -1: [[1, (1)], [1, 1]]
# -1  0: [[1, 1], [(1), 1]]
# -1 -1: [[1, 1], [1, (1)]]
'''

see source code in scipy ndimage.binary_dilation (27/02/2023)
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1

    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 1, brute_force)
'''

ad_d00 = binary_dilation(ad, structure=struct,
                         origin=(0, 0)).astype(int)
ad_d01 = binary_dilation(ad, structure=struct,
                         origin=(0, -1)).astype(int)
ad_d10 = binary_dilation(ad, structure=struct,
                         origin=(-1, 0)).astype(int)
ad_d11 = binary_dilation(ad, structure=struct,
                         origin=(-1, -1)).astype(int)

print(f"ad:\n{ad}\n\nad_d00:\n{ad_d00}\n\nad_d01:\n{ad_d01}\n\nad_d10:\n{ad_d10}\n\nad_d11:\n{ad_d11}")
