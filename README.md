# amoeba
This module is a basic implementation of the simplex (Nelder-Mead) method of function minimzation.

It is base ond [this version of the algorithm](http://www.scholarpedia.org/article/Nelder-Mead_algorithm), and hopefully it is clean and simple enough to allow any specific customization.

There is just one class, `Amoeba`, which incorporates the simplex and the necessary methods to drive it.

## Class initialization

The function to minimize, `U`, should take a numpy array as its argument and return a real number. The simplex itself is then an array of such arrays. The points (simplex vertices) don't have to be 1D arrays, e.g. `U` might have matrix arguments. What matters is that they can be added and multiplied by numbers; the 0th axis of the simplex numbers vertices, the remaining axes available for internal structure of the specific problem.

Usually when there are `d` problem parametrs (the length of argument of `U`), the simplex has `d+1` points. But because of specific embedding one might want different dimensions of the simplex and the embedding space. Whatever shape the initial simplex has, it will be accepted, but explicit dimension can be specified (with keyword `dim`) to remove the co-dimension warning. If no simplex is specified, the default is `[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]` and it's higher-dimensional analogons.

For most iterations, `U` will be evaluated once or twice on speific points, but for some operations it has to be called on all points. It then makes sense to have ready a vectorize/parallel/optimized version, `U_opt`, that takes the whole simplex array as the argument. With the initial simplex `sim0`, the object is then created via (`f` is required, the others optional)

`ameba = Amoeba(f=U0, fmap=U_opt, sim=sim0, dim=d)`

Because they are costly, the setup operations have to be called separately with `ameba.prep()` - this evaluates `U` (or `U_opt`) on all vertices, sorts them (see `ord` below), and finds the (pseudo-centroid). The object is then ready for minimization proper.

## Class attributes

The simplex itself is stored in `ameba.simplex` with the values of the function in `ameba.values`. The vertices are modified in place as the algorithm proceeds, but they are not switched: the current order (from minimal to maximal) of values (and therefore vertices) is stored in `ameba.ord`, so that `ameba.values[ameba.ord]` is sorted.

The parameters (and their default values) of the algorithm itself are</br>
`ameba.alpha = 1.0` - for the reflection case (see the link to the algorithm details [above](#amoeba) </br>
`ameba.beta = 0.5` - for contraction,</br>
`ameba.gamma = 2.0` - for expansion,</br>
`ameba.delta = 0.5` - for shrinking.

The dimension, number of vertices and current iteration are `ameba.dim`, `ameba.len` and `ameba.iterations`, respectively.

## Class methods

The method to be called before the main loop is `ameba.prep()`, which is just a shorthand for:</br>
`ameba.eval()` - evaluates (with the optimized fmap if available) the function on all vertices,</br>
`ameba.order()` - performs argsort, to update the `ord` attribute,</br>
`ameba.center()` - calculates the pseudo-centroid (without division, and using all points).

The main method is which performs a single case check and propagation is
`ameba.step(end='\n', quiet=False, trial=False)`</br>
It returns a tuple `(s,i)`, where `s` specifies the algorithm case, and `i` is the rank to which the previous worst point moved (0 means it landed in a new minimum). The cases are (see also [below](#examples)):</br>
1 expansion (new minimum),</br>
2 reflection (new minimum),</br>
3 reflection (same minimum),</br>
4 contraction,</br>
5 shrinking.</br>

The `trial` keyword allows to check which algorithm case is next, but does not perform the corresponding transformation. To suppress messages use `quiet=True`. The `end` keyword allows to get rid of the trailing newline of the output, so more information can be printed. This is illustrated in the [example2.py](/example2.py) script.

The simplex can be saved and loaded using hdf5 files with `ameba.save(fname)` and `ameba.load(fname)` with string file name.

The current best point is accessible through the `ameba.best` property, and there is a naive method `ameba.size()` that gives the maximal distance (norm) of the best vertex from all the other vertices.

## Examples

The most basic usage is given in [example1.py](/example1.py), where the Rosenbrock function's minimum is found (there are two, so the result depends on the random initial condition). There is a fixed limit of steps, and a breaking condition on the simplex's size. The output uses colored kanji, for faster recognition of the algorithm case, based on observation of how well the process is going they were chosen as:</br>
🔵 夏 - (case 1) when not just a reflection, but expanded reflection found a new minimum,</br>
🟢 春 - (Case 2) when ordinary reflection yielded a new minimum,<br>
🟠 秋 - (case 3) when reflection improved the current worst point considerably (but the minimum stayed the same),</br>
⚪ 冬 - (case 4) when the worst point was improved by internal/external contraction (but not beyond the second worst).

When the worst happens and shrinking is required (case 5), 🔴 is used, like so:</br>
<img alt="output of example1.py" src="/example1.png" width=480></br>
The numbers after the arrow show the rank to which the old worst vertex jumped, after the case-specific transformation (0 means it replaced the previous best minimum).

The second example [example2.py](/example2.py) shows how to use the output of `ameba.step()` to display more information, like the relative improvement, and spread in the values. It has dimension 20, and also shows the potential gain in using `fmap`.

- [ ] Example with visualization, as Jupyter notebook, in preparation.
