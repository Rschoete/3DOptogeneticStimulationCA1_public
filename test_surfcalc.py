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

a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0],
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

print(f"a:\n {a}\n\n a_o:\n {a_o}\n\n a_f:\n {a_f}\n\n {a_f_f}")


ad = np.zeros(np.array(a.shape)+[2, 1])
ad[1:-1, 0:-1] = a
ad_d = binary_dilation(
    ad, structure=np.ones((3, 3)), iterations=1).astype(int)
ad_f = enclosed_by_four(ad_d & atleast_one(ad))
ad_f_f = enclosed_by_four(binary_dilation(ad, structure=np.ones((3, 3)),
                                          iterations=1).astype(int)) & atleast_one(ad)
print(f"ad: \n {ad}\n\n a_o: \n {ad_d}\n\n a_f: \n {ad_f}\n\n{ad_f_f}\n\n")
