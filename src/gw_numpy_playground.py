import numpy as np


a = [[1,2],
     [3,4]]
a=np.array(a)
assert a.shape == (2,2)

b = [[[1,2], [2,3]],

     [[3,4], [4,5]]]

b = np.array(b)
assert b.shape == (2,2,2)

c = [[[[1,2],
       [3,4]],
      [[4,5],
       [6,7]]],
     [[[7,8],
       [9,10]],
      [[10,11],
       [12,13]]]]

c = np.array(c)
assert c.shape == (2,2,2,2)


c_sum_2 = np.sum(c, axis=2)

print(c_sum_2)

c_sum_3 = np.sum(c, axis = 3)

print(c_sum_3)

c_norm_2 = np.linalg.norm(c, axis=2)
c_norm_3 = np.linalg.norm(c, axis=3)
