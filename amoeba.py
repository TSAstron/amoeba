import numpy
import h5py
try:
  from termcolor import colored
except ImportError:
  print('Warning, could not import the termcolor module. Switching to standard output.')
  colored = lambda q, c: q

class Amoeba():
    def __init__(self, f=None, fmap=None, simplex=None, dim=None):
        """ f to minimize, fmap is the optional vectorize/parallel version """
        self.f = f
        self.fmap = fmap
        self.simplex = numpy.array([[0],[1]], dtype='float64') if simplex is None else simplex
        self.get_dims(dim)
        self.alpha = 1.0 # reflection
        self.beta  = 0.5 # contraction
        self.gamma = 2.0 # expansion
        self.delta = 0.5 # shrinking
        self.iterations = 0
        self.values = numpy.zeros(self.len, dtype='float64')
        self.ord = numpy.arange(self.len)
    def get_dims(self, dim=None):
        self.dim = numpy.prod( self.simplex.shape[1:] ) if dim is None else dim
        self.len = self.simplex.shape[0] # len(self.simplex) # simplices of lower dimension allowed
        self.di = int(numpy.log10(self.len)) + 1
        if self.len != self.dim+1:
            print(f'Warning: the simplex has co-dimension {self.dim+1-self.len}.')
    def save(self, fname):
        file = h5py.File(fname, 'w')
        for k in file.keys():
            del file[k]
        file.create_dataset('simplex', data=self.simplex)
        file.create_dataset('values', data=self.values)
        file.close()
    def load(self, fname):
        file = h5py.File(fname, 'r')
        self.simplex = file['simplex'][()]
        self.values = file['values'][()]
        file.close()
        self.get_dims()
        self.order()
        self.center()
        print('Simplex loaded. Checking...\r', end='', flush=True)
        try:
            b, w = self.ord[[0,-1]]
            bf, wf = self.f(self.simplex[b]), self.f(self.simplex[w])
            print(f"Simplex loaded. Best and worst agreement with current f: {bf-self.values[b]:.2e}, {wf-self.values[w]:.2e}.")
        except:
            print("Simplex loaded, but not functions specified for evaluation.")
    def order(self):
        """ find the sorting order/indices for the current values of f (no evaluation) """
        self.ord = numpy.argsort(self.values)
    def center(self):
        """ calculate the "centroid": use ALL vertices and do not divide by length -
        the division happens later due to optimization, same for exclusion of the best vertex """
        self.centroid = self.simplex.sum(axis=0)
    @property
    def best(self):
        """ the index of the best (minimal) vertex """
        return self.simplex[self.ord[0]]
    def eval(self, start=0, lvl=0):
        """ evaluate f on all vertices, beginning at 'start'. The order is the current self.ord,
        so start=0 begins at the minimal vertex """
        if self.fmap is None: # when no optimized version is give, it's a simple loop
            pret = ' │ ' if lvl>0 else ''
            print(pret+'fmap not specified, proceeding with ordinary loop')
            for i, o in enumerate(self.ord[start:]):
                print( f"{pret}{i+1:{self.di}d} of {self.len-start}\r", end='', flush=True)
                self.values[o] = self.f(self.simplex[o])
            if lvl>0: print(f' └ {self.len-start} of {self.len-start}', flush=True)
        else: # Thi parallel/vectorized version currently ignores the 'start' parameter
            print( pret+'Parallel evaluation in process...\r', end='', flush=True)
            self.values[:] = self.fmap( self.simplex ) 
            print( ' └ Parallel evaluation finished.', flush=True)
    def prep(self, lvl=0):
        """ basic preparation of the initial simplex: evaluate f on all vertices, sort them, find the "centroid" """
        self.eval(lvl=lvl)
        self.order()
        self.center()
    def shrink(self, fac=None, start=1):
        """" Shrink thesimplex towards the ebst vertex, normally applies from the second point onwards """
        best = self.best
        fac_u = self.delta if fac is None else fac
        for o in self.ord[start:]: 
            self.simplex[o] = best + fac_u*(self.simplex[o]-best)
    def reject(self, quiet=False, trial=False):
        """ wrapper around the failed contraction case: info + shrink + cleanup preparation """
        print( ('contraction' if quiet else ' ') + colored('rejected','red') + ' → shrinking' )
        if not trial:
            self.shrink()
            self.prep(lvl=1)
            self.iterations += 1
        return 5
    def accept(self, w, xx, ff, possibilite):
        """ accept the new vertex, incrementally update the "centroid" """
        self.centroid += (xx - self.simplex[w])
        self.simplex[w] = xx
        self.values[w] = ff
        self.order()
        self.iterations += 1
        return possibilite
    def locate(self, ff):
        """ find where the new value lands in the current order """
        ii = 0
        while ff > self.values[self.ord[ii]]:
            ii += 1
        return ii

    def step(self, end='\n', quiet=False, trial=False):
        """ Main function - determine the case (expansion, reflection, contraction, shrink)
        and take the step - unless 'trial' is set to True """
        w = self.ord[-1] # position of the worst
        x_0 = (self.centroid - self.simplex[w])/(self.len-1) # reflection point, centroid of the sub-simplex
        x_r = x_0 + self.alpha*(x_0-self.simplex[w])
        f_r = self.f(x_r)
        if f_r < self.values[self.ord[0]]: # better than current best -> try expanding
            x_e = x_0 + self.gamma*(x_r - x_0)
            f_e = self.f(x_e)
            if f_e < f_r: # even better -> accept expanded as new 1st # greedy minimalization version
                if not quiet: print( colored('夏','cyan') + ' expansion accepted as best'.ljust(33) + '→ ' + colored(f'{0:{self.di}d}','cyan'), end=end, flush=True )
                return( 0 if trial else self.accept(w, x_e, f_e, 0), 0)
            else: # expansion does not help -> accept reflected (still a better mininum)
                if not quiet: print( colored('春','green') + ' reflection accepted as best'.ljust(33) + '→ ' + colored(f'{0:{self.di}d}','green'), end=end, flush=True )
                return( 1 if trial else self.accept(w, x_r, f_r, 1), 0)
        elif f_r < self.values[self.ord[-2]]: # better than second worst -> accept reflection (same minimum, improved maximum)
            ii = self.locate(f_r)
            if not quiet: print( colored('秋', 'yellow') + ' reflection accepted'.ljust(33) + f'→ {ii:{self.di}d}', end=end, flush=True )
            return( 2 if trial else self.accept(w, x_r, f_r, 2), ii)
        else: # try contracting
            if f_r < self.values[w]: # better than the worst, so contract outside
                if not quiet: print( colored('冬','white') + ' external contraction', end='', flush=True)
                x_c = x_0 + self.beta*(x_r - x_0)
                f_c, sz = self.f(x_c), '¹ '
                if f_c > f_r: # standard contraction would fail, try improving
                    x_c = x_0 + 0.5*self.beta*(x_r - x_0)
                    f_c, sz = self.f(x_c), '² '
                if f_c <= f_r: # accept contraction
                    ii = self.locate(f_c)
                    if not quiet: print( sz+'accepted  → '+ colored(f'{ii:{self.di}d}','green' if ii==0 else None), end=end,  flush=True)
                    return( 3 if trial else self.accept(w, x_c, f_c, 3), ii)
                else:
                    return( self.reject(quiet, trial), self.len)
            else: # reflection is the worst, so contract inside
                if not quiet: print( colored('冬','white') + ' internal contraction', end='', flush=True )
                x_c = x_0 + self.beta*(self.simplex[w] - x_0)
                f_c, sz = self.f(x_c), '¹ '
                if f_c >= self.values[w]: # standard contraction would fail, try improving
                    x_c = x_0 + 0.5*self.beta*(self.simplex[w] - x_0)
                    f_c, sz = self.f(x_c), '² '
                if  f_c < self.values[w]: # better than the worst, accept contraction
                    ii = self.locate(f_c)
                    if not quiet: print( sz+'accepted  → '+ colored(f'{ii:{self.di}d}','green' if ii==0 else None), end=end, flush=True )
                    return( 4 if trial else self.accept(w, x_c, f_c, 4), ii)
                else:
                    return( self.reject(quiet, trial), self.len)
                  
    def size(self):
        """ simplex size estimate: max distance of the best vertex from the others; warning: could be costly! """
        best = self.best
        return max( numpy.linalg.norm(self.simplex[o]-best) for o in self.ord[1:] )
