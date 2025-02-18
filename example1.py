from amoeba import Amoeba
import numpy

def rosen(X):
    """ the Rosen function for arbitrary number of arguments """
    return (1-X[:-1])@(1-X[:-1]) + 100*(X[1:]-X[:-1]*X[:-1])@(X[1:]-X[:-1]*X[:-1])

# initial simplex: 4 varibales, so 5 points
sim0 = 10*numpy.random.random((4+1,4))

# create the crawler with the Rosen function, no optimized fmap, automatic dimension
# sim0 is not copied internally and WILL be modified in the process!
ameba = Amoeba(f=rosen, simplex=sim0)

# preparation: initial values, order and centroid
ameba.prep()

# the basic loop looks like this
# break criterion based on size, which is the scale of the simplex
# so the actual minimum is expected to lie within that distance from the current best

x_tol = 5e-16

for _ in range(1000):
    ameba.step()
    if ameba.size() <= x_tol:
        break
else:
  print(f'Warning, 1000 steps taken, but the size is above the threshold ({x_tol}).')

print(f'The value of f at x = {ameba.best} is {ameba.values[]}.')
