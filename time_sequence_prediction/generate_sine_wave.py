import math
import numpy as np
import cPickle as pickle
T = 20
L = 1000
N = 100
np.random.seed(2)
x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x / 1.0 / T).astype('float64')
pickle.dump(y, open('traindata.pkl', 'wb'))

