'''
| Filename    : __init__.py
| Description : Handy decorators and context managers for improved REPL experience.
| Author      : Pushpendre Rastogi
| Created     : Thu Oct 29 19:43:24 2015 (-0400)
| Last-Updated: Sun Apr 24 09:20:58 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 224
'''
import collections
import contextlib
import time
import numpy
import random
import print_hook
import sys
from lev import lev
import itertools
import os
try:
    import ipdb as pdb
except ImportError:
    import pdb


def print_indent_fn(text):
    if len(text) > 0:
        indent = print_indent_fn.indent
        white_space = (2 * indent * ' ')
        return text.replace('\n', '\n' + white_space)
    else:
        return text


def print_indent_and_redirect_to_file(text):
    text = print_indent_fn(text)
    print_indent_and_redirect_to_file.ofh.write(text)
    return text


def setup_print_indent(ofh=None):
    print_indent_fn.indent = 0
    print_indent_and_redirect_to_file.ofh = ofh
    if print_hook.PrintHook().already_started:
        print_hook.PrintHook().stop()

    setup_print_indent.printhook = print_hook.PrintHook().start(
        func=(print_indent_fn
              if ofh is None
              else print_indent_and_redirect_to_file),
        override='stdout')
    return setup_print_indent.printhook


def increase_print_indent():
    try:
        print_indent_fn.indent += 1
    except AttributeError:
        setup_print_indent()
        increase_print_indent()


def decrease_print_indent():
    print_indent_fn.indent -= 1

DISABLE_TICTOC = False


@contextlib.contextmanager
def tictoc(msg):
    ''' Simplify the addition of timing and profiling instructions in python
    code. Use this context manager as:

    with tictoc("Description"):
        Some long computation.

    The output would be as follows where XXX is the time taken in seconds.
    Started Description
    Completed Description in XXXs
    '''
    if not DISABLE_TICTOC:
        t = time.time()
        increase_print_indent()
        print "Started", msg
    yield
    if not DISABLE_TICTOC:
        decrease_print_indent()
        print
        print "Completed", msg, "in %0.1fs" % (time.time() - t)


class DecoratorBase(object):

    ''' The `functools.wraps` function takes a function used in a decorator
    and adds the functionality of copying over the function name, docstring,
    arguments list, etc.

    But for class-style decorators, @wrap doesn't do the job. This base class
    solves the problem by proxying attribute calls over to the function that
    is being decorated.

    Copied from: stackoverflow.com/questions/308999/what-does-functools-wraps-do
    '''
    func = None

    def __init__(self, func):
        self.__func = func

    def __getattribute__(self, name):
        if name == "func":
            return super(DecBase, self).__getattribute__(name)

        return self.func.__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "func":
            return super(DecBase, self).__setattr__(name, value)

        return self.func.__setattr__(name, value)


class announce(DecoratorBase):

    ''' Decorate a function with this to announce entrance.
    Use this decorator as:

    @announce()
    def func():
        print 1

    The output would be
    Started func
    1
    '''

    def __init__(self, *args, **kwargs):
        return

    def __call__(self, f):
        def runtime_wrapper(*args, **kwargs):
            print "Started", f.__name__
            return f(*args, **kwargs)
        return runtime_wrapper


@contextlib.contextmanager
def announce_ctm(task):
    increase_print_indent()
    print "Started", task
    yield
    decrease_print_indent()
    print
    print "Finished", task


class reseed(DecoratorBase):

    ''' Reseed both `random` and `numpy.random` with a particular seed before
    starting a function. This is useful to quickly pin down a case where a
    particular function is not consistently working even though we set the
    seed explicitly at the beginning of a program.
    '''

    def __init__(self, seed, reset=True):
        self.seed = seed
        self.reset = reset
        return

    def __call__(self, f):
        def runtime_wrapper(*args, **kwargs):
            numpy.random.seed(self.seed)
            random.seed(self.seed)
            val = f(*args, **kwargs)
            if self.reset:
                numpy.random.seed(self.seed)
                random.seed(self.seed)
            return val
        return runtime_wrapper


@contextlib.contextmanager
def reseed_ctm(seed, reset=True):
    ''' A context manager for seeding a particular groups of statements.
    and then resetting the seed after executing the statements (by default)
    '''
    state = random.getstate()
    numpy.random.seed(seed)
    random.seed(seed)
    yield
    if reset:
        numpy.random.seed(seed)
        random.seed(seed)
        random.setstate(state)


def _top_frame(frame):
    if frame.f_back is None:
        return frame
    else:
        return _top_frame(frame.f_back)


def _call_pdb(_sig, _frame):
    import code
    d = dict(_frame=_frame, top_frame=_top_frame)
    return code.InteractiveConsole(d).interact(
        "Entering python shell. Press Ctrl-d to resume execution.")


def disable_debug_support():
    debug_support._debug = False


def enable_debug_support():
    debug_support._debug = True


@contextlib.contextmanager
def debug_support(capture_ctrl_c=True):
    ''' Drop into
    '''
    if not hasattr(debug_support, '_debug'):
        debug_support._debug = True
    if debug_support._debug:
        import signal
        signal.signal(signal.SIGUSR1, _call_pdb)
        if capture_ctrl_c:
            signal.signal(signal.SIGINT, _call_pdb)
            pass
    try:
        yield
    except:
        sys_exc_info = sys.exc_info()
        if debug_support._debug:
            import traceback
            traceback.print_exc()
            pdb.post_mortem(sys_exc_info[2])
        else:
            raise sys_exc_info[0], sys_exc_info[1], sys_exc_info[2]


# http://www.dalkescientific.com/writings/diary/archive/2005/04/20/tracing_python_code.html
# https://pymotw.com/2/trace/index.html#module-trace
# https://pymotw.com/2/sys/tracing.html
# Set the system's trace function, which allows you to implement a Python source
# code debugger in Python. The function is thread-specific; for a debugger to
# support multiple threads, it must be registered using settrace() for each
# thread being debugged.
# Trace functions should have three arguments: frame, event, and arg.
# frame is the current stack frame.
# event is a string: 'call', 'line', 'return', 'exception', 'c_call',
# 'c_return', or 'c_exception'.
# arg depends on the event type.

# The trace function is invoked (with event set to 'call') whenever a new local
# scope is entered; it should return a reference to a local trace function to be
# used that scope, or None if the scope shouldn't be traced.
# The local trace function should return a reference to itself (or to another
# function for further tracing in that scope), or None to turn off tracing in
# that scope.

# The events have the following meaning:
# 'call'       # A function is called (or some other code block entered). The
# global trace function is called; arg is None; the return value specifies the
# local trace function.
# 'line'       # The interpreter is about to execute a new line of code or
#                re-execute the condition of a loop.
# 'return'     # A function (or other code block) is about to return. The local
# trace function is called; arg is the value that will be returned, or None if
# the event is caused by an exception being raised. The trace function's return
# value is ignored.
# 'exception'  # An exception has occurred. The local trace function is called;
# arg is a tuple (exception, value, traceback); the return value specifies the
# new local trace function.
# 'c_call'     # A C function is about to be called. This may be an extension
# function or a built-in. arg is the C function object.
# 'c_return'   # A C function has returned. arg is the C function object.
# 'c_exception'# A C function has raised an exception. arg is the C function
# object.
# Note that as an exception is propagated down the chain of callers, an
# 'exception' event is generated at each level.
# For more information on code and frame objects, refer to The standard
# type hierarchy.


# @contextlib.contextmanager
# def step_through():
#     def fn(
#     sys.settrace(fn)
#     yield
#     sys.settrace(lambda _, __, ___: None)


class Namespace(collections.MutableMapping):

    """Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.
    """
    __hash__ = None

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        for name, value in self.__dict__.iteritems():
            arg_strings.append('%s=%r' % (name, value))
        if self.__name is None:
            return '%s(%s)' % (type_name, ', '.join(arg_strings))
        else:
            return '%s(%s, %s)' % (
                type_name, self.__name, ', '.join(arg_strings))

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            assert len(args) == 1
            self.__name = args[0]
        else:
            self.__name = None
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __ne__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return not (self == other)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def update_and_append_prefix(self, ns, prefix=None):
        ''' Add keys from given namespace object. If prefix is None this
        delegates to the defualt update method.
        Params
        ------
        ns     : The namespace object.
        prefix : The prefix.
        Returns
        -------
        self
        '''
        if prefix is None:
            self.update(ns)
        else:
            for k in ns:
                self[prefix + k] = ns[k]
        return self

    def copy_invariant_is_suffix(
            self, invariant, prefix_source, prefix_dest, glue='_'):
        ''' Copy a property from [prefix_source][glue][invariant] to
        [prefix_dest][glue][invariant]
        '''
        setattr(self,
                prefix_dest + glue + invariant,
                getattr(self, prefix_source + glue + invariant))
        return

    def copy_invariant_is_prefix(
            self, invariant, src_suffix, dest_suffix, glue='_'):
        '''Copy property from
        [invariant][glue][src_suffix] -> [invariant][glue][dest_suffix]
        '''
        setattr(self,
                invariant + glue + dest_suffix,
                getattr(self, invariant + glue + src_suffix))
        return


def flatten(lol):
    ''' Convert a nested list to a flat list
    Params
    ------
    lol : List of List
    Returns
    -------
    l : list
    '''
    l = []
    for e in lol:
        if isinstance(e, list) or isinstance(e, tuple):
            l.extend(flatten(e))
        else:
            l.append(e)
    return l


def batch_list(lst, n=1):
    l = len(lst)
    for ndx in range(0, l, n):
        yield lst[ndx:min(ndx + n, l)]


class NameSpacer(
        collections.MutableMapping,
        collections.MutableSequence,
        collections.MutableSet):

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return self.obj.__repr__()

    def __eq__(self, other):
        return self.obj == other

    def __ne__(self, other):
        return not (self.obj == other)

    def __contains__(self, key):
        return key in self.obj

    # Items are for ['']
    def __getitem__(self, key):
        return self.obj.__getitem__(key)

    def __setitem__(self, key, value):
        self.obj.__setitem__(key, value)

    # Attrs are for .
    def __getattr__(self, key):
        return getattr(self, 'obj').__getitem__(key)

    def __delitem__(self, key):
        self.obj.__delitem__(key)

    def __iter__(self):
        return self.obj.__iter__()

    def __len__(self):
        return self.obj.__len__()

    def __add__(self, right_obj):
        return NameSpacer(self.obj.__add__(right_obj.obj)
                          if isinstance(right_obj, NameSpacer)
                          else self.obj.__add__(right_obj))

    def insert(self, i, e):
        return self.obj.insert(i, e)

    def add(self, e):
        self.obj.add(e)

    def discard(self, e):
        self.obj.discard(e)


def namespacer(obj):
    return NameSpacer(obj)


def deep_namespacer(nested_dict):
    ''' Many times we have a nested hierarchy of dictionaries.
    But sometimes for agility we don't want to use the string
    based indexing syntax. This function converts a deep nested
    dictionary into nested NameSpacer objects
    Params
    ------
    nested_dict : A dict of dicts and other values.

    '''
    return namespacer(
        dict((k, deep_namespacer(v) if isinstance(v, dict) else v)
             for k, v in nested_dict.iteritems()))


def sample_from_list(lst, samples, return_generator=False):
    ''' Sample `samples` many points from a lst that are spaced evenly apart.
    If samples is a float then we may return a little less than exactly specified
    since it is not possible to return the exact percentage requested.
    Params
    ------
    lst : A list of objects.
    samples : int or float
      In case samples is a floating point it should be between 0 and 1.
      Then samples represents the fraction of the list to return. Otherwise
      samples represents the number of samples to return.
    Returns
    -------
    A subsampled list with
    '''
    l = len(lst)
    if isinstance(samples, float):
        assert 0 <= samples <= 1
        samples = int(l * samples)
    else:
        assert isinstance(samples, int)
    if samples == 0:
        return []
    elif samples >= l:
        return lst
    else:
        # Find out the step size to take.
        step_size = int(l / samples)
        if not return_generator:
            return lst[:samples * step_size:step_size]
        else:
            return (lst[i] for i in xrange(0, samples * step_size, step_size))


def validate_np_array(
        value, name=None, _max=1e6, _min=-1e6, mean=1e6, silent_fail=False,
        describe=0):
    if isinstance(value, numpy.ndarray):
        msg_template = '%s breached %s limit, limit=%f, value=%f\n'
        for limit_type, v, l in [('max', value.max(), _max),
                                 ('abs-of-mean', abs(value.mean()), mean),
                                 ('mean-of-abs', numpy.absolute(value).mean(), mean)]:
            assert v < l, msg_template % (str(name), limit_type, l, v)
            if describe:
                print 'Parameter %s limit_type %s, limit=%f, value=%f' % (
                    str(name), limit_type, l, v)
        for limit_type, v, l in [('min', value.min(), _min)]:
            assert v > l, msg_template % (str(name), limit_type, l, v)
            if describe:
                print 'Parameter %s limit_type %s, limit=%f, value=%f' % (
                    str(name), limit_type, l, v)
    else:
        if not silent_fail:
            raise NotImplementedError
        else:
            pass
    return


def sort_dictionary_by_values_in_descending_order(d):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


def get_tokenizer():
    from pattern.en import tokenize
    return lambda x: tokenize(x)[0].split(' ')


def pipeline_tokenizer():
    ''' This function can be called from the cmd line to
    tokenize files from command line.
    '''
    tknzr = get_tokenizer()
    for row in sys.stdin:
        print ' '.join(tknzr.tokenize(row.strip()))


def pipeline_dictionary(pattern_tokenize=0, lowercase=0):
    ''' This function is called from the commandline to extract a dictionary
    from a file after tokenizing it.
    '''
    tokenizer = (get_tokenizer()
                 if pattern_tokenize
                 else
                 (lambda x: x.split(' ')))
    d = collections.defaultdict(int)
    for row in sys.stdin:
        row = row.strip()
        for token in tokenizer(row):
            if lowercase:
                token = token.lower()
            d[token] += 1
    for k, v in sort_dictionary_by_values_in_descending_order(d):
        print k


def process_columns(f, *args, **kwargs):
    ''' process each row passed in through stdin.
    It prints the processed row to stdout.
    Params
    ------
    f     : The function to apply to each column
    *args : If args is non empty then it specifies the columnd to be processed.
    '''
    get = lambda k, v: (kwargs[k] if k in kwargs else v)
    ifs = get('ifs', None)
    ofs = get('ofs', '\t')
    ors = get('ors', '\n')
    for row in sys.stdin:
        for idx, col in enumerate(row.split(ifs)):
            idx += 1
            if len(args) == 0 or idx in args:
                col = f(col)
            sys.stdout.write(col)
            sys.stdout.write(ofs)
        sys.stdout.write(ors)
    return


def _warning(message, category=UserWarning, filename='', lineno=-1):
    print >> sys.stderr, message
    return

import warnings
warnings.showwarning = _warning


def warn(msg):
    warnings.warn(msg)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


def put(a, b, idx, replace=False):
    ' Put a:list into b:list at idx. '
    assert isinstance(a, list)
    assert isinstance(b, list)
    assert isinstance(idx, int)
    assert idx >= 0 and idx <= len(b)
    if idx == 0:
        if replace:
            return b + a[1:]
        else:
            return b + a
    elif idx == len(b):
        if replace:
            raise NotImplementedError("Cant replace at b[len(b)]")
        return a + b
    else:
        if replace:
            return b[:idx] + a + b[idx + 1:]
        else:
            return b[:idx] + a + b[idx:]


def nonparametric_signed_test_for_significance(arr1, arr2):
    ''' Given two numpy arrays of equal length containing samples of
    statstics that we want to reliably tell apart, we perform the signed
    permutation test, which is a simplified variant of the paired permutation,
    test.

    The basic idea is that we can tell whether the mean of the distribution
    that gave rise to the data in arr1 and arr2 are truly different irrespective
    of the distributions that gave rise to arr1, and arr2.

    We chose the null hypothesis to be that the two distributions have the same
    mean (and nothing else, i.e. we dont assume a parametric form of the
    distribution or anything else) and then we find whether under random swaps
    of elements of arr1 and arr2 the absolute value of the average difference
    remains lower than the absolute value of average difference when no swapping
    is done.

    Note that one of the permutations is the identity permutation which means
    that the count has to be at least 1.

    Note: One desirable property of this procedure is that it is scale
    independent since all we are doing is checking for inequalities even if both
    v1, v2 are scaled by some constant k. The results would be the same.

    See axon.cs.byu.edu/Dan/478/assignments/permutation_test.php for details.
    Params
    ------
    arr1, arr2 : Two numpy 1d arrays.
    '''
    assert arr1.ndim == 1
    assert arr2.ndim == 1

    def absolute_value_of_mean(v):
        return float(abs((v).mean()))

    v = arr1 - arr2
    base_aad = absolute_value_of_mean(v)
    n = 0
    t = 0
    for signs in itertools.product([-1, 1], repeat=v.shape[0]):
        sign_aad = absolute_value_of_mean(numpy.array(signs) * v)
        # Note: That we do [>=] instead of [>] is theoretically important.
        n += (sign_aad >= base_aad)
        t += 1
    return float(n) / t


def flatdict_iterator(fh):
    '''
    A flatdict is a file that contains data like the following:

    index: 0
    text: sentence1
    partof: train

    index: 1
    text: sentence2
    partof: test

    This is a flat representation of a dictionary that is human readable.
    This function iterates over the entries in such a flatdict file.
    Params
    ------
    fh : The file handle.

    Returns
    -------
    An iterator to iterate over the entries in a flat file.
    '''
    d = {}
    for row in fh:
        row = row.strip()
        if row == '':
            yield d
            d = {}
        else:
            row = row.split(': ')
            key = row[0]
            val = ': '.join(row[1:])
            d[key] = val


@contextlib.contextmanager
def numpy_print_ctm(**kwargs):
    ''' See
    docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.set_printoptions.html

    The important keys are:
    precision: Number of digits of precision for floating point output. (default 8)
    threshold: numel in array which triggers sumarization.
    linewidth: number of characters per line for inserting line breaks.
    suppress: suppress printing small floats using scientific notation.
    '''
    if len(kwargs) > 0:
        orig_opt = dict(numpy.get_printoptions())
        numpy.set_printoptions(**kwargs)
    yield
    if len(kwargs) > 0:
        numpy.set_printoptions(**orig_opt)


def randomized_check_grad(func, grad, x0, verbose=0, quota=20, tol=1e-8,
                          return_decision=True, eps=1e-6):
    ''' Numerically checks the directional derivative of a high dimensional function
    along points located in randomly chosen directions away from x0.

    This is essentially a faster, randomized version of `scipy.optimize.check_grad`
    which incidentally is useless for high dimensional functions because it changes
    parameters one at a time, so it needs to compute the gradient `P` times where
    `P` is the number of parameters.

    Istead we just check that the numerical directional derivative along random
    directions (y-x0) equals the theoretical value grad'(y - x0) for random y chosen
    from a coordinatewise uniform [x0-eps, x0+eps] distribution.

    Params
    ------
    func    : The function.
    grad    : The theoretical gradient.
    x0      : The point around which `grad` needs to be checked.
    quota   : The number of random directions to evaluate. (default 10)
              The error along each random direction must be less than tol.
              TODO: Relatively? Absolutely?
    tol     : The tolerance for error (default 1e-8)
    eps     : The deviation in parameter values of the random points around x0.
    verbose : (default 0)

    Returns
    -------
    Either a boolean or a floating point indicating the maximum error along a
    random direction.
    (err < tol) if return_decision else err

    Usage Example:
    --------------
    >>> print randomized_check_grad(
            lambda x: np.dot(x.T, x), lambda x: 2 * x, x0 = np.ones((10,)))
    True
    >>> print randomized_check_grad(
            lambda x: np.dot(x.T, x), lambda x: x, x0 = np.ones((10,)))
    False
    '''
    p = x0.shape[0]
    fx0 = func(x0)
    dY = ((numpy.random.rand(p, quota) - 0.5) * eps)
    Y = x0[:, numpy.newaxis] + dY
    numerical_dfY = numpy.array(
        [func(Y[:, idx]) - fx0 for idx in range(quota)])
    gradient_at_x0 = grad(x0)
    theoretical_dfY = numpy.array(
        [numpy.dot(gradient_at_x0, dY[:, idx])
         for idx in range(quota)])
    errors = numpy.abs(numerical_dfY - theoretical_dfY)
    if verbose:
        with numpy_print_ctm(precision=2):
            print 'individual_errors\n', errors
    err = max(errors)
    return ((err < tol)
            if return_decision
            else err)


def ensure_dir(f, verbose=False, treat_as_dir=False):
    ''' Copied from stackoverflow.com/questions/273192/
    in-python-check-if-a-directory-exists-and-create-it-if-necessary
    Params
    ------
    f       : File name whose parent directories are guaranteed to exist
              at the end of this operation. (Upto OS Errors and Exceptions)
    treat_as_dir: Is the input file really a directory.
    verbose : (default False)
    '''
    d = (f
         if treat_as_dir
         else os.path.dirname(f))
    if not os.path.exists(d):
        os.makedirs(d)
        if verbose:
            print 'Created directory', d


def majority(lst):
    return sort_dictionary_by_values_in_descending_order(
        collections.Counter(lst))[0][0]


def crossval(n, k, shuffle=True):
    from sklearn.cross_validation import KFold
    return KFold(n=n, n_folds=k, shuffle=shuffle)


class cache_to_disk(DecoratorBase):

    def __init__(self, output_fn='cache_to_disk.pkl', hash_fnc=None):
        ''' This decorator caches the output of a function to a shelf on disk.

        Params
        ------
        output_fn: The name of the shelf file in which the outputs are stored.
            If None
        hash_fnc : The function used for hashing the inputs of the decorated
            function. If this function is not provided then the default
            hash function in python is used over the inputs.
        '''
        self.output_fn = output_fn
        self.hash_fnc = hash_fnc
        return

    def __call__(self, f):
        def runtime_wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return runtime_wrapper


def confidence_interval_of_mean_with_unknown_variance(obs, alpha=0.9, sample_contains_all_of_population=False):
    '''
    Params
    ------
    obs : An array of observations.
    Returns
    -------
    The mean and a confidence interval around it.
    '''
    obs = numpy.array(obs).astype('float64')
    assert obs.ndim == 1
    if obs.shape[0] == 0:
        raise ValueError("obs should not be an empty array")
    if obs.shape[0] == 1:
        mean = obs[0]
        interval = (mean, mean)
        return (mean, interval)

    sample_mean = obs.mean()
    standard_deviation = (obs.std(ddof=0)
                          if sample_contains_all_of_population
                          else obs.std(ddof=1))
    if standard_deviation < 1e-10:
        return (sample_mean, (sample_mean, sample_mean))
    n = obs.shape[0]
    standard_error = standard_deviation / numpy.sqrt(n)

    from scipy.stats import t

    return (sample_mean,
            t.interval(alpha, n - 1, loc=sample_mean, scale=standard_error))

from mpmath import mp, mpf
mp.dps = 256

# The mp_ functions can be 20 times slower !!


def mp_log1mexp(x):
    return mp.log(mpf(1) - mp.exp(x))


def log1mexp(x):
    if x >= 0:
        raise ValueError
    elif x < -1e10:
        return numpy.log1p(-numpy.exp(x))
    elif x <= -1e-6:
        return numpy.log(-numpy.expm1(x))
    else:
        return x + numpy.log(numpy.expm1(-x))
