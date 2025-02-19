from amoeba import Amoeba
import numpy

def rosen(X):
    """ the Rosenbrock function for arbitrary number of arguments """
    return (1-X[:-1])@(1-X[:-1]) + 100*(X[1:]-X[:-1]*X[:-1])@(X[1:]-X[:-1]*X[:-1])

def r_map(sim):
    X1 = 1 - sim[:,:-1]
    X2 = sim[:,1:] - sim[:,:-1]*sim[:,:-1]
    return ( X1*X1 + 100*X2*X2 ).sum(axis=1)

# initial simplex: 4 varibales, so 5 points
sim0 = 2*numpy.random.random((20+1,20))


import time
t0 = time.time()
for x in sim0:
    (rosen(x))
t1 = time.time() - t0
print(f'Normal evaluation: {t1}s')

t0 = time.time()
(r_map(sim0))
t2 = time.time() - t0

print(f'Vectorized evaluation: {t2}s')
print(f'Speedup: {t1/t2:.2f}x')

input('Press enter to continue.')

# create the crawler with the Rosenbrock function, no optimized fmap, automatic dimension
# sim0 is not copied internally and WILL be modified in the process!
ameba = Amoeba(f=rosen, fmap=r_map, simplex=sim0)

# preparation: initial values, order and centroid
ameba.prep()

# the basic loop looks like this
# break criterion based on size, which is the scale of the simplex
# so the actual minimum is expected to lie within that distance from the current best

# define some display format of function values
form = lambda x: f'{x: .4f}' if abs(x) >= 0.0001 else f'{x: .2e}'

# the initial minimal value
min00 = ameba.values[ameba.ord[0]]
print(f'The initial value of f at x = {ameba.best} is {min00:.4e}.\n')

# ameba.len is the number of points so this loop more or less 
# goes over the whole simplex 100 times
# stops after the first shrinking
for i in range(2000*ameba.len):
    # ameba.di is the number of digits, whichis useful for padding
    print(f"{i+1:{ameba.di+2}d}", end=' ', flush=True)
    # the minimum before the step is taken
    min0 = ameba.values[ameba.ord[0]]
    # get the status, and output the default message without the trailing
    # newline so other info can be appended
    status, _ = ameba.step(end='')
    # indices of the best and worst points
    b, w = ameba.ord[0], ameba.ord[-1]
    rel_gain = (min0-ameba.values[b])/abs(min0)
    # Δ is the (value) spread between best and worst points, δ is the relative improvement for the new minimum
    print( f"   [new worst: {form(ameba.values[w])} Δ={ameba.values[w]-ameba.values[b]:.3e}" + (f" new best: {form(ameba.values[b])} δ={rel_gain:.2e}]" if rel_gain>0 else "]") )
    if status==5: # status 5 means that shrinking happened
        break

print(f"\nTotal gain: {form(ameba.values[b]-min00)}, relative change δ={(min00-ameba.values[b])/abs(min00):.3e}")

print(f'After {ameba.iterations} iterations, the value of f at x = {ameba.best} is {ameba.values[ameba.ord[0]]:.4e}.')
