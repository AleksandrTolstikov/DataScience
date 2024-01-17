import numpy as np

import numpy as np

w = np.zeros((16))

D = np.array([
   [0,1,1,0,
    1,0,0,1,
    1,0,0,1,
    0,1,1,0,],
   [1,1,1,1,
    1,0,0,1,
    1,0,0,1,
    1,1,1,1,],
   [0,1,1,0,
    1,0,0,1,
    1,1,1,1,
    1,0,0,1,],
   [0,1,1,0,
    1,1,1,1,
    1,0,0,1,
    1,0,0,1,],
])

Y = np.array([1,1,0,0,0,0])

α =  0.2 
β = -0.4 
σ = lambda x: 1 if x > 0 else 0

def f(x, _w):
    s = β + np.sum(x @ _w)
    return σ(s)

def train(w, D, Y):
    _w = w.copy()
    for x, y in zip(D, Y):
        w += α * (y - f(x, w)) * x
    return (w != _w).any()

while train(w, D, Y):
    print(w)

D = np.array([
   [0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,], 
   [0,1,1,0,
    1,0,0,1,
    1,0,0,1,
    0,1,1,0,],
   [1,1,1,1,
    1,0,0,1,
    1,0,0,1,
    1,1,1,1,],
   [0,1,1,0,
    1,0,0,1,
    1,1,1,1,
    1,0,0,1,],
   [0,1,1,0,
    1,1,1,1,
    1,0,0,1,
    1,0,0,1,],
])

for x in D:
    print(x, f(x, w))
