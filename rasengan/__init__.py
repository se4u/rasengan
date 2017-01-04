'''
| Filename    : __init__.py
| Description : Reusable decorators and context managers for expeditious development.
| Author      : Pushpendre Rastogi
| Created     : Thu Oct 29 19:43:24 2015 (-0400)
| Last-Updated: Wed Jan  4 14:09:09 2017 (-0500)
|           By: System User
|     Update #: 479
'''
from __future__ import print_function
from . import print_hook
from . import sPickle
import BaseHTTPServer
import base64
import collections
import contextlib
import functools
import itertools
import numpy
import os
import pdb
import random
import re
import six
import string
import sys
import time
try:
    import termcolor
except ImportError:
    pass
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from .lev import lev  # pylint: disable=import-error
except ImportError:
    pass
try:
    import html.entities
except ImportError:
    pass
try:
    from unidecode import unidecode
except ImportError:
    pass
try:
    import scipy.sparse
    import scipy.misc
except ImportError:
    pass

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


def setup_print_indent(ofh=None, override='stdout'):
    print_indent_fn.indent = 0
    print_indent_and_redirect_to_file.ofh = ofh
    if print_hook.PrintHook().already_started:
        print_hook.PrintHook().stop()

    setup_print_indent.printhook = print_hook.PrintHook().start(
        func=(print_indent_fn
              if ofh is None
              else print_indent_and_redirect_to_file),
        override=override)
    return setup_print_indent.printhook


def increase_print_indent(override='stdout'):
    try:
        print_indent_fn.indent += 1
    except AttributeError:
        setup_print_indent(override=override)
        increase_print_indent(override=override)


def decrease_print_indent():
    print_indent_fn.indent -= 1

DISABLE_TICTOC = False


@contextlib.contextmanager
def tictoc(msg, override='stdout'):
    ''' Simplify the addition of timing and profiling instructions in python
    code. Use this context manager as:

    with tictoc("Description"):
        Some long computation.

    The output would be as follows where XXX is the time taken in seconds.
    Started Description
    Completed Description in XXXs
    '''
    stream = (sys.stdout if override == 'stdout' else sys.stderr)
    if not DISABLE_TICTOC:
        t = time.time()
        increase_print_indent(override=override)
        print("Started", msg, file=stream)
    yield
    if not DISABLE_TICTOC:
        decrease_print_indent()
        print("\nCompleted", msg, "in %0.1fs" % (time.time() - t), file=stream)


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
            return super(DecoratorBase, self).__getattribute__(name)

        return self.func.__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "func":
            return super(DecoratorBase, self).__setattr__(name, value)

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
            print("Started", f.__name__)
            return f(*args, **kwargs)
        return runtime_wrapper


@contextlib.contextmanager
def announce_ctm(task):
    increase_print_indent()
    print("Started", task)
    yield
    decrease_print_indent()
    print()
    print("Finished", task)


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
            # Fixed because of github.com/se4u/neural_wfst/issues/1
            # Fix based on stackoverflow.com/questions/14503751
            # how-to-write-exception-reraising-code-thats-compatible-
            # with-both-python-2-and-python-3
            six.reraise(*sys_exc_info)


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
        for name, value in self.__dict__.items():
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
    ''' Convert a nested list to a flat list. NOTE: This is not recursive !!
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


class NaNList(collections.MutableSequence):

    ''' A list which allows indexing with NaN. When indexed with NaN
    the list returns NaN. The list itself can contain NaNs.
    '''

    def __init__(self, *args):
        self.obj = list(args)

    def __repr__(self):
        return self.obj.__repr__()

    def __eq__(self, other):
        return self.obj == other

    def __ne__(self, other):
        return not (self.obj == other)

    def __contains__(self, val):
        return val in self.obj

    def __getitem__(self, key):
        try:
            return self.obj.__getitem__(key)
        except:
            return float('nan')

    def __setitem__(self, key, value):
        self.obj.__setitem__(key, value)

    def __delitem__(self, key):
        self.obj.__delitem__(key)

    def __iter__(self):
        return self.obj.__iter__()

    def __len__(self):
        return self.obj.__len__()

    def insert(self, i, e):
        return self.obj.insert(i, e)


class NameSpacer(
        collections.MutableMapping,
        collections.MutableSequence,
        collections.MutableSet):

    '''
    TODO: What does this do?
    '''

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
             for k, v in nested_dict.items()))


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
            return (lst[i] for i in range(0, samples * step_size, step_size))


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
                print('Parameter %s limit_type %s, limit=%f, value=%f' % (
                    str(name), limit_type, l, v))
        for limit_type, v, l in [('min', value.min(), _min)]:
            assert v > l, msg_template % (str(name), limit_type, l, v)
            if describe:
                print('Parameter %s limit_type %s, limit=%f, value=%f' % (
                    str(name), limit_type, l, v))
    else:
        if not silent_fail:
            raise NotImplementedError
        else:
            pass
    return


def sort_dictionary_by_values_in_descending_order(d):
    return sorted(list(d.items()), key=lambda x: x[1], reverse=True)


def get_tokenizer():
    from pattern.en import tokenize  # pylint: disable=import-error,no-name-in-module
    return lambda x: tokenize(x)[0].split(' ')


def pipeline_tokenizer():
    ''' This function can be called from the cmd line to
    tokenize files from command line.
    '''
    tknzr = get_tokenizer()
    for row in sys.stdin:
        print(' '.join(tknzr.tokenize(row.strip())))


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
    for k, _ in sort_dictionary_by_values_in_descending_order(d):
        print(k)


class TokenMapper(object):
    '''
    t2i = Token to index map
    During construction call it.
    '''

    def __init__(self, *args):
        self.t2i = {}
        self.vocab_size = 0
        self.i2t = None
        self.final = False
        if len(args):
            self.__call__(args)

    def __len__(self):
        return len(self.t2i)

    def __call__(self, tokens):
        ''' Call this to get the index of tokens.
        '''
        l = []
        for tok in tokens:
            i = None
            try:
                i = self.t2i[tok]
            except KeyError as e:
                if self.final:
                    raise e
                else:
                    i = self.vocab_size
                    self.t2i[tok] = i
                    self.vocab_size += 1
            l.append(i)
        return l

    def finalize(self, max_tok=None):
        if max_tok is None:
            max_tok = len(self.t2i)
        self.t2i['<BOS>'] = max_tok
        self.max_tok = len(self.t2i)
        self.i2t = dict((a, b) for (b, a) in self.t2i.iteritems())
        self.final = True
        return self.t2i['<BOS>']

    def __getitem__(self, index):
        '''
        Call this to get the token at index.
        '''
        try:
            return self.i2t[index]
        except TypeError:
            return [self[i] for i in index]
        except KeyError:
            ct = index % self.max_tok
            pt = ((index - ct) / self.max_tok - 1)
            try:
                return (self.i2t[pt], self.i2t[ct])
            except KeyError:
                import pdb
                pdb.set_trace()


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


def _warning(message, category=UserWarning, filename=None, lineno=-1):
    print(message, file=sys.stderr)
    return

import warnings
warnings.showwarning = _warning


def warn(msg):
    warnings.warn('Warning: ' + str(msg))


@contextlib.contextmanager
def warn_ctm(msg):
    warn(msg)
    yield
    return


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
            print('individual_errors\n', errors)
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
    d = os.path.expanduser(f
                           if treat_as_dir
                           else os.path.dirname(f))
    if not os.path.exists(d):
        os.makedirs(d)
        if verbose:
            print('Created directory', d)


def majority(lst):
    return sort_dictionary_by_values_in_descending_order(
        collections.Counter(lst))[0][0]


def crossval(n, k, shuffle=True):
    from sklearn.cross_validation import KFold
    return KFold(n=n, n_folds=k, shuffle=shuffle)


class cache_to_disk(object):

    def __init__(self, shelf_fn='cache_to_disk.pkl', hash_fnc=None):
        ''' This decorator caches the output of a function to a shelf on disk.
        TODO: The current implementation does not handle keyword arguments
        to function.
        Params
        ------
        shelf_fn:
            The name of the shelf file in which the outputs are stored.
        hash_fnc : x -> string
            The function used for hashing the inputs of the decorated
            function. The hash_fnc must return a string.
            If this function is not provided then we just convert the
            input to string.
        '''
        self.shelf_fn = shelf_fn
        self.hash_fnc = (lambda x: str(x)
                         if hash_fnc is None
                         else hash_fnc)

    def __call__(self, f):
        import shelve
        try:
            shelf = shelve.open(self.shelf_fn)
        except Exception as e:
            print (
                e, 'If db type could not be determined then probably you are using an existing file.')
            raise Exception(e)

        def runtime_wrapper(*args):
            key = str(self.hash_fnc(args))
            try:
                return shelf[key]
            except KeyError:
                val = f(*args)
                shelf[key] = val
                return val

        runtime_wrapper.shelf = shelf
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

    from scipy.stats import t  # pylint: disable=no-name-in-module

    return (sample_mean,
            t.interval(alpha, n - 1, loc=sample_mean, scale=standard_error))

try:
    # The mp_ functions can be 20 times slower !!
    from mpmath import mp, mpf
    mp.dps = 256
except ImportError:
    pass
else:
    def mp_log1mexp(x):
        return mp.log(mpf(1) - mp.exp(x))  # pylint: disable=no-member


def log1mexp(x):
    if x >= 0:
        raise ValueError
    elif x < -1e10:
        return numpy.log1p(-numpy.exp(x))
    elif x <= -1e-6:
        return numpy.log(-numpy.expm1(x))
    else:
        return x + numpy.log(numpy.expm1(-x))


def pivot_lol_by_col(lol, col):
    ''' Pivot a lol(list of lists) to a dict of lol
    keyed by the column.

    Example:
    >>> lol = [[1, 2, 4], [3, 2], [4, 5]]
    >>> print pivot_lol_by_col(lol, 1)
    {2: [[1 4], [3,]], 5: [[4]]}
    '''
    d = collections.defaultdict(list)
    for l in lol:
        d[l[col]].append(l[:col] + l[col + 1:])
    return dict(d)


def pivot_key_to_list_map_by_item(klmap):
    d = collections.defaultdict(list)
    for k in klmap:
        for e in klmap[k]:
            d[e].append(k)
    return dict(d)


def around(s, at=0, bound=20):
    if isinstance(bound, int):
        start_bound = stop_bound = bound
    else:
        start_bound, stop_bound = bound
    start = max(0, at - start_bound)
    stop = min(len(s), at + stop_bound)
    return s[start:stop]


def html_entity_to_unicode_impl(m):
    text = m.group(0)
    if text[:2] == "&#":
        # character reference
        try:
            return (int(text[3:], 16)
                    if text[:3] == "&#x"
                    else int(text[2:]))
        except ValueError:
            pass
    else:
        # named entity
        try:
            text = (html.entities.html5[text[1:]])  # pylint: disable=no-member
        except KeyError:
            pass
    return text


def html_entity_to_unicode(text):
    ''' Convert html code point to unicode.

    Copied from http://stackoverflow.com/questions/57708
    convert-xml-html-entities-into-unicode-string-in-python
    '''
    return re.sub("&#?\w+;", html_entity_to_unicode_impl, text)  # pylint: disable=anomalous-backslash-in-string


def unicode_to_utf8_hex(c, prefix='%', fmt="X"):
    assert sys.version_info.major >= 3
    return "".join(
        [(prefix + format(e, fmt))
         for e
         in c.encode("utf-8")])


def wiki_encode_url(url):
    assert sys.version_info.major >= 3
    return re.sub(r'[^\x00-\x7F]',
                  lambda c: unicode_to_utf8_hex(c.group(0)),
                  url).replace('#', '%23')


def flatfile_to_dict(fn, key_col=0, total_col=2):
    assert total_col == 2, 'Not Implemented'
    assert key_col == 0, 'Not Implemented'
    d = dict()
    with open(fn) as f:
        for row in f:
            row = row.strip().split()
            d[row[key_col]] = row[1]
    return d


class OrderedDict_Indexable_By_StringKey_Or_Index(collections.MutableMapping):

    ''' An ordered map from *Non-Integer keys* to values that does not support deletion.
    We can index from either the string key OR the location of the string.
    '''

    def __init__(self, *args, **kwargs):
        self._store = {}
        self._idx2key = []
        for (a, b) in args:
            self.__setitem__(a, b)
        for (a, b) in kwargs.iteritems():
            self.__setitem__(a, b)
        return

    def __contains__(self, key):
        return key in self._store

    def getkey(self, key_idx):
        assert isinstance(key_idx, int)
        return self._idx2key[key_idx]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._store[self._idx2key[key]]
        else:
            return self._store[key]

    def __setitem__(self, key, value):
        if isinstance(key, int) and key >= 0:
            raise Exception('Positive Integer valued keys are prohibited!')
        if key not in self._store:
            self._idx2key.append(key)
        self._store[key] = value
        pass

    def __delitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._idx2key)

    def __len__(self):
        return len(self._idx2key)


def urlsafe_b64encode_pipe():
    for row in sys.stdin:
        print(base64.urlsafe_b64encode(row))


def urlsafe_b64decode_pipe():
    for row in sys.stdin:
        sys.stdout.write(base64.urlsafe_b64decode(row.strip()))


class _TcpStdIOShim_handler(BaseHTTPServer.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'
    # If set to 'HTTP/1.1', the server will permit HTTP persistent connections;
    # however, the server must then include an accurate Content-Length header
    # (using send_header()) in all of its responses to clients.

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "json")
        self.end_headers()
        return

    def do_GET(self):
        self.do_HEAD()
        if self.server.stdio_proc.verbose:
            print('self.path=', self.path)
        self.wfile.write(
            base64.urlsafe_b64encode(
                self.server.stdio_proc(
                    base64.urlsafe_b64decode(self.path[1:]))))
        return


class _TcpStdIOShim_proc(object):

    def __init__(self, expect_proc, output_until, verbose=False):
        self.expect_proc = expect_proc
        self.output_until = output_until
        self.verbose = verbose

    def __call__(self, request):
        self.expect_proc.send(request + '\n'
                              if request[-1] != '\n'
                              else request)
        ret = self.expect_proc.expect_exact(self.output_until)
        if self.verbose:
            print('request=', request, 'ret=', ret,
                  'self.expect_proc.before=', self.expect_proc.before)
        return self.expect_proc.before


class TcpStdIOShim(object):

    ''' USAGE:
    python -c 'from rasengan import TcpStdIOShim as T; T(verbose=True).execute()'
    '''

    def __init__(self, port=13579, cmd='cat', output_until='\n', verbose=False):
        import pexpect
        self.stdio_proc = _TcpStdIOShim_proc(
            pexpect.spawn(cmd), output_until, verbose)
        self.http_daemon = BaseHTTPServer.HTTPServer(
            ('', port), _TcpStdIOShim_handler)
        self.http_daemon.stdio_proc = self.stdio_proc
        self.verbose = verbose
        if verbose:
            print('Server Configured')

    def execute(self):
        if self.verbose:
            print('Server Running')
        try:
            self.http_daemon.serve_forever()
        except KeyboardInterrupt:
            if self.verbose:
                print('Received Keyboard interrupt')
        finally:
            self.http_daemon.server_close()
            self.stdio_proc.expect_proc.kill(15)

def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer

PUNCT_CHAR = frozenset(''.join(chr(e) for e in range(
    33, 48) + range(58, 65) + range(91, 97) + range(123, 127)))
REGEX_SPECIAL_CHAR = frozenset(r'[]().-|^{}*+$\?')
PUNCT_MATCH_REGEX = re.compile(
    '([%s])'%(''.join(
        ('\\%s'%e if e in REGEX_SPECIAL_CHAR else e)
        for e in PUNCT_CHAR)))

@memoize
def _clean_text_construct_regex(runs):
    ''' Remove multiple runs of punctuations. E.g.
    convert ...... -> .
    '''
    repeater = '{%d,}' % runs
    l = ([e + repeater
          for e in PUNCT_CHAR
          if e not in REGEX_SPECIAL_CHAR]
         + ['\\' + e + repeater
            for e
            in REGEX_SPECIAL_CHAR])
    s = '(%s)' % ('|'.join(l))
    return re.compile(s)


def _clean_text_sub_fn(obj):
    return obj.group(0)[0]


def clean_text(text,
               tr=('?[]{}', '.()()'),
               remove_punct_runs=3):
    ''' Remove unicode and substitute bad characters, based on
    tuples of substitutions.
    '''
    if isinstance(text, str):
        text = text.decode('utf-8')
    text = unidecode(text).translate(string.maketrans(*tr))
    if remove_punct_runs:
        run_rex = _clean_text_construct_regex(remove_punct_runs)
        return run_rex.sub(_clean_text_sub_fn, text)
    else:
        return text


class pklflow_ctx(object):

    def __init__(self, in_fn, out_fn):
        import argparse
        arg_parser = argparse.ArgumentParser(description='')
        arg_parser.add_argument('--in_fn', default=in_fn, type=str)
        arg_parser.add_argument('--out_fn', default=out_fn, type=str)
        self.args = arg_parser.parse_args()
        self.ns = Exception("NS")
        self.ns.out_data = None
        with open(self.args.in_fn) as f:
            self.ns.data = pickle.load(f)

    def __enter__(self):
        'NOTE: The return value of __enter__ is received by as'
        return self.ns

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            with open(self.args.out_fn, 'wb') as f:
                pickle.dump(self.ns.out_data, f)
        else:
            print("Not Saving any data, since exception occurred. "
                  "It is possible to implement strategies for incrementally "
                  "saving data.")


@memoize
def sentence_segmenter_tokenizer():
    import nltk
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer._params.abbrev_types.update(  # pylint: disable=protected-access,no-member
        ['e.j', 'u.s', 'b.p', 'pres', 'gov', '1a', 'fmr', 'rev', 'pfc', 'lieut',
         'lt', 'ex', 'sec', 'atty', 'gen', 'pvt', 'rep', 'def', 'capt', 'jr',
         'fr', 'mr', 'ms', 'hon', 'dr', 'gen', 'cmdr', 'cpl', 'sgt', 'det', 'ca',
         'reb', 'shri', 'sir', 'lieut', 'cpt'])
    tokenizer._params.collocations.update(  # pylint: disable=protected-access,no-member
        [('lt', 'governor'),
         ('mt', 'everest'),
         ('j.d', 'hayworth'),
         ('j.d', 'baldwin')])
    # Get rid of the junk punkt training.
    tokenizer._params.ortho_context = collections.defaultdict(int)
    return tokenizer


def sentence_segmenter(para):
    ''' Segment the paragraph into sentences.
    Returns the span_starts and span_ends as a list of tuples.
    '''
    tokenizer = sentence_segmenter_tokenizer()
    return tokenizer.span_tokenize(para)  # pylint: disable=no-member


def tokens_in_tokenization_corresponding_to_a_span(sent, start, end, tokens):
    ''' Consider the sentence 'I love candy.'
    Let's say our existing span of interest is (start=3, end=8); but our
    tokenization is [I, love, candy, .]
    We will return [1, 3] as the smallest interval of tokens that overlaps
    our segment of interest. This interval excludes endpoint.

    >> f = tokens_in_tokenization_corresponding_to_a_span
    >> f('I love candy and apple.', 7, 12, ['I', 'love', 'candy', 'and', 'apple', '.'])
    (2, 3)
    '''
    assert end > start
    space_till_start = sent[:start].count(' ')
    space_till_end = sent[:end].count(' ')
    sent = sent.replace(' ', '')
    start = start - space_till_start
    end = end - space_till_end
    t_start = 0
    start_tok = 0
    for t_idx, t in enumerate(tokens):
        t_end = t_start + len(t)
        # Now find the index of highest t_start that is less than or equal to
        # start.
        if t_start <= start:
            start_tok = t_idx
        # Find the index of smallest t_end that is higher than or equal to end.
        if t_end >= end:
            return (start_tok, t_idx + 1)
        t_start = t_end
    # This return statement is only reached if we cant find an end.
    return (start_tok, len(tokens))


def get_referents(canonical_mention_sentence_id,
                  canonical_mention_tokens,
                  canonical_mention_token_end_idx,
                  tokenized_sentences,
                  pronomial_coref=True):
    ''' Consider a list of sentences like:
    - Howard was glad that, like so many other doubters, he was proving Hugh
      wrong.
    - Howard also mentioned that he saw his SIRIUS coworker, [Martha Stewart],
      last night when he was dining with his daughter.
    - Howard said Martha came over to his table and requested some
      pointers for her satellite

    The tokens in brackets are known to
    correspond to a know entity, clearly other mentions of 'Martha' and 'she'
    and 'her' also refer to the same entity. This function marks all
    coreferent tokens of this type. We use the following heuristic, any
    tokens that match with the canonical mention tokens are coreferent.
    Any pronouns that appear after the canonical mention are also coreferent.

    We return a list of (sentence_id, token_id)

    >> from rasengan import get_referents as g;
    >> s= ['Howard was glad that , like so many other doubters, he was proving Hugh wrong'.split(),
           'Howard also mentioned that he saw his SIRIUS coworker Martha Stewart last night when he was dining with his daughter .'.split(
               ),
           'Howard said Martha came over to his table and requested some pointers for her satellite'.split()]
    >> v = g(1, ['Martha', 'Stewart'], 11, s)
    >> print [s[a][b] for [a, b] in v]
    ['Martha', 'Stewart', 'Martha', 'her']
    '''

    PRONOUN_TO_GENDER = dict(him=0, his=0, he=0, she=1, her=1, hers=1)
    # from gender import gender_detector, PRONOUN_TO_GENDER
    cmt = [e.lower() for e in canonical_mention_tokens]
    ret = []
    unlikely_gender = {}
    for (sent_idx, sent) in enumerate(tokenized_sentences):
        for tok_idx, tok in enumerate(sent):
            tok = tok.lower()
            if tok in cmt:
                ret.append((sent_idx, tok_idx))
            if pronomial_coref:
                if ((sent_idx < canonical_mention_sentence_id
                     or (sent_idx == canonical_mention_sentence_id
                         and tok_idx < canonical_mention_token_end_idx))
                        and tok in PRONOUN_TO_GENDER):
                    unlikely_gender[PRONOUN_TO_GENDER[tok]] = None
                if ((sent_idx > canonical_mention_sentence_id
                     or (sent_idx == canonical_mention_sentence_id
                         and tok_idx >= canonical_mention_token_end_idx))
                        and tok in PRONOUN_TO_GENDER
                        and PRONOUN_TO_GENDER[tok] not in unlikely_gender):
                    ret.append((sent_idx, tok_idx))
    return ret


def reshape_conll(parse):
    ID = NaNList()
    W = NaNList()
    Tc = NaNList()
    Tf = NaNList()
    P = NaNList()
    R = NaNList()
    for (i, w, _1, tc, tf, _2, p, r, _3, _4) in parse:
        ID.append(int(i) - 1)
        W.append(w)
        Tc.append(tc)
        Tf.append(tf)
        P.append(int(p) - 1)
        R.append(r)
    return (ID, W, Tc, Tf, P, R)


class open_wmdoe(object):

    '''wmdoe = Write Mode, Delete on Exception.
    Open File in write mode, and in case of exceptions delete the file.
    '''

    def __init__(self, fn, extra='b'):
        self.f = open(fn, 'w' + extra)

    def __enter__(self):
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            pass
        else:
            print('Deleting file', self.f.name, file=sys.stderr)
            self.f.close()
            os.remove(self.f.name)


def entity_descriptors(referents, conll_parse, debug_print=False):
    ''' Get the tokens that describe the tokens mentioned in
    /referents/ using the provided conll_parse, which is a list
    of lists.
    '''
    # Initialize the descriptors.
    ETS = referents
    B = {}
    D = {}
    CONVERGED = False
    (ID, W, Tc, Tf, P, R) = reshape_conll(conll_parse)
    # NOTE: This pattern of growing a set to convergence by
    # applying rules should be repeatable. It would probably
    # need standardization of what we are growing.
    # Grow descriptors till convergence.
    while not CONVERGED:
        OLD_LEN_BD = len(B) + len(D)
        for (i, w, tc, tf, p, r) in itertools.izip(
                ID, W, Tc, Tf, P, R):
            if ((p in ETS and r == 'appos')
                    or (p in D and r in ['acomp', 'nn'])
                    or (P[p] in D and r in ['pobj', 'pcomp'])
                    or (p in B and r in ['pobj', 'dobj'])
                    or (P[p] in B and R[p] in ['pobj', 'dobj']) and r == 'conj'):
                D[i] = True
            if ((i in ETS and r in ['nsubj', 'nsubjpass'])
                    or (i in ETS and r in ['poss', 'advmod'])):
                D[p] = True
            if (Tc[p] == 'VERB' and r == 'dobj' and p in D):
                B[p] = True
        NEW_LEN_BD = len(B) + len(D)
        CONVERGED = (NEW_LEN_BD == OLD_LEN_BD)
        pass
    if debug_print and len(referents):
        print(markup_tokens(
            W, dict([(_, 'red') for _ in D]
                    + [(_, 'green') for _ in referents])), '\n',
              file=sys.stderr)
    return D


def markup_tokens(tokens, idx_to_color, sep=' '):
    string_builder = []
    for idx, token in enumerate(tokens):
        try:
            string_builder.append(
                str(termcolor.colored(token, idx_to_color[idx])))
        except KeyError:
            string_builder.append(token)
    return sep.join(string_builder)


def uniq_c(iterable, sort=False):
    ret = collections.Counter(iterable).items()
    if sort:
        ret.sort(key=lambda e: e[1], reverse=True)
    return ret


def deduplicate_unhashables(lst):
    '''
    >> deduplicate_unhashables([1, 1, 3, 2, 3, 3, 4])
    [1, 2, 3, 4]
    '''
    lst = sorted(lst)
    ret = [lst[0]]
    for e in lst[1:]:
        if e != ret[-1]:
            ret.append(e)
    return ret


def groupby(fn, mode='r', predicate=None, yield_iter=False):
    predicate = ((lambda x: x != '\n')
                 if predicate is None
                 else predicate)
    # Since we can't reliably check whether a file is a bzipfile
    # or a plain file, or a gzip file by just checking whether
    # fn is instance of `file` therefore the best way to distinguish
    # between a file object and a string seems to be to check whether
    # the object has a `close` attribute or not.
    # I could be overtly restrictive by checking instances of each
    # type, file, gzip.GzipFile, bzip.BzipFile etc. but that seems
    # to be too cumbersome.
    grouper = (itertools.groupby(fn, predicate)
               if hasattr(fn, 'close')
               else itertools.groupby(open(fn, mode=mode), predicate))
    for k, v in grouper:
        if k:
            yield (v if yield_iter else list(v))


class GrayCombinatorialCounter(object):

    def __init__(self, lim=(2, 3, 3), return_jumps=True, add_beginning_state=False):
        self.lim = lim
        self.state = [0] * len(lim)
        self._raise = False
        self.carry = [False] * len(lim)
        self.carry_from = list(lim)
        assert not (
            return_jumps and add_beginning_state), "Jumps can't have beginning state"
        self.return_jumps = return_jumps
        self.add_beginning_state = add_beginning_state

    def pos_to_update(self, ppos):
        if ppos < 0:
            raise StopIteration
        if self.state[ppos] != ((self.carry_from[ppos] - 1) % self.lim[ppos]):
            return ppos
        elif self.state[ppos] == ((self.carry_from[ppos] - 1) % self.lim[ppos]):
            self.carry[ppos] = True
            self.carry_from[ppos] = self.state[ppos]
            return self.pos_to_update(ppos - 1)
        elif (self.state[ppos] == self.carry_from[ppos]
              and self.carry[ppos]):
            self.carry[ppos] = False
            return ppos
        else:
            raise Exception()

    def update(self, p2u):
        self.state[p2u] = ((self.state[p2u] + 1)
                           % self.lim[p2u])
        return self

    def next(self):
        if self.add_beginning_state:
            self.add_beginning_state = False
            return list(self.state)
        else:
            p2u = self.pos_to_update(len(self.lim) - 1)
            oldval = self.state[p2u]
            self.update(p2u)
            newval = self.state[p2u]
            return ((p2u, oldval, newval)
                    if self.return_jumps
                    else list(self.state))

    def __iter__(self):
        return self


def exp_normalize(arr):
    deno = scipy.misc.logsumexp(arr)
    return [numpy.exp(e - deno) for e in arr]


def force_open(fn, mode='r'):
    try:
        return open(fn, mode)
    except IOError:
        os.makedirs(os.path.dirname(fn))
        open(fn, 'a').close()
    return open(fn, mode)


def ngramatize(l, n=2, bos='<BOS>'):
    if n == 1:
        return l
    l = [bos] * (n - 1) + l
    return [tuple(l[i:i + n]) for i in range(len(l) - (n - 1))]



class NamespaceLite(collections.MutableMapping):
    __hash__ = None

    def __repr__(self):
        return self._name

    def __init__(self, pfx, **kwargs):
        self._name = pfx + ''.join('.%s~%s'%(a, str(b)) for (a,b) in kwargs.iteritems())
        for key, val in kwargs.iteritems():
            setattr(self, key, val)

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

class OpenOverride(object):
    def __init__(self, pfx, open_fnc):
        self.pfx = pfx
        self.open_fnc = open_fnc
        return

    def __call__(self, name, mode='r', buffering=None):
        assert isinstance(name, str)
        if mode == 'r':
            try:
                return self.open_fnc(name, mode=mode, buffering=buffering)
            except IOError:
                return self.open_fnc(
                    os.path.join(self.pfx, name), mode=mode, buffering=buffering)
        return self.open_fnc(name, mode=mode, buffering=buffering)

def csr_mat_builder(iterator, shape, dtype='float32', verbose=0):
    data = []
    indices = []
    indptr = [0]
    for row_idx, row in enumerate(iterator):
        if verbose > 0 and row_idx % verbose == 0:
            print('@row:', row_idx, file=sys.stderr)
        for j in sorted(row):
            indices.append(j)
            data.append(row[j])
        indptr.append(len(indices))
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape,
                                   dtype=dtype)


class ConstantList(collections.MutableSequence):
    def __init__(self, e, length):
        self.e = e
        self.l = length

    def __repr__(self):
        return '[%s]*%d'%(str(self.e), self.l)

    def __eq__(self, other):
        return (isinstance(other, ConstantList) and self.e == other.e and \
                self.l == other.l)

    def __contains__(self, e):
        return e == self.e

    def __getitem__(self, i):
        if i < self.l:
            return self.e
        raise IndexError('list index out of range')

    def __setitem__(self, _, __):
        raise NotImplementedError()

    def __delitem__(self, _):
        raise NotImplementedError()

    def __iter__(self):
        return itertools.repeat(self.e, self.l)

    def __len__(self):
        return self.l

    def insert(self, _, __):
        raise NotImplementedError()

def save_sparse_matrix(file_handle, m):
    if scipy.sparse.isspmatrix_coo(m):
        return numpy.savez(file_handle, row=m.row, col=m.col, data=m.data,
                           shape=m.shape, type='coo')
    elif scipy.sparse.isspmatrix_csr(m) or scipy.sparse.isspmatrix_csc(m):
        type = ('csc' if scipy.sparse.isspmatrix_csc(m) else 'csr')
        return numpy.savez(file_handle, data=m.data, indices=m.indices,
                           indptr=m.indptr, shape=m.shape,
                           type=type)
    else:
        raise NotImplementedError(
            'Invalid type: '+type(m)+'. We have only implemented saving CSR,\n'
            'CSC and COO because those are the most efficient formats to save.')

def load_sparse_matrix(filename):
    d = numpy.load(filename)
    assert 'shape' in d and 'type' in d
    if d['type'] == 'coo':
        return scipy.sparse.coo_matrix(
            (d['data'], (d['row'], d['col'])), shape=d['shape'])
    elif d['type'] in ['csr', 'csc']:
        constructor = (scipy.sparse.csc_matrix
                       if d['type'] == 'csc'
                       else scipy.sparse.csr_matrix)
        return constructor((d['data'], d['indices'], d['indptr']),
                           shape=d['shape'])
    else:
        raise NotImplementedError()

def farthest_non_empty_row(lilmat):
    assert scipy.sparse.isspmatrix_lil(lilmat)
    for i in xrange(lilmat.shape[0]-1, -1, -1):
        if len(lilmat.rows[i]) > 0:
            return i+1
    return 0



class StreamingArrayMaker(object):
    '''
    An object to hold multiple sparse data matrices with corresponding rows, along
    with their names. The update_1 method minimizes the number of method
    calls since the data for a document is passed in one shot for all arrays.
    '''
    def __init__(self, categories, pfx=None, default_shape=(2000000, 2000000)):
        self.categories = categories
        if pfx is not None:
            self.arr_list = None
            self.ent_map = None
            self.tm_list = None
            self.load_from_pfx(pfx)
        else:
            self.arr_list = [scipy.sparse.lil_matrix(default_shape,
                                                     dtype='int8')
                             for e
                             in self.categories]
            self.ent_map = TokenMapper()
            self.tm_list = [TokenMapper() for e in self.categories]
        return

    @staticmethod
    def largest_non_empty_column(lilmat):
        return max(max(lilmat.rows[i])
                   for i
                   in xrange(lilmat.shape[0])
                   if len(lilmat.rows[i]))

    @staticmethod
    def tokenize(s):
        return [e for e in re.split(PUNCT_MATCH_REGEX, s) if e != '']

    def update_1(self, docid_str, cat_data):
        '''
        NOTE: This updation is fast only if the docid_str are
        received in sorted order during repeated calls
        '''
        row_id = self.ent_map([docid_str])[0]
        for i, data in enumerate(cat_data):
            cols = self.tm_list[i](self.tokenize(data))
            self.arr_list[i][row_id, cols] = 1
        return

    def finalize(self):
        max_row = max(farthest_non_empty_row(e) for e in self.arr_list)
        for i in range(len(self.categories)):
            max_col = self.largest_non_empty_column(self.arr_list[i])
            # Cut down the size of matrices
            self.arr_list[i] = self.arr_list[i][:max_row, :max_col]
            self.arr_list[i] = self.arr_list[i].tocoo()  # tocsr()
            print(self.arr_list[i].shape)
        return

    def save_to_pfx(self, pfx):
        self.ent_map.finalize()
        for i in range(len(self.tm_list)):
            self.tm_list[i].finalize()
        fn = pfx+'_tokenmap.pkl'
        with tictoc('Saving TokenMaps to %s'%fn):
            pickle.dump([self.ent_map, self.tm_list], open(fn, 'wb'),
                        protocol=-1)
        for cat, arr in zip(self.categories, self.arr_list):
            fn = '%s_%s.npz'%(pfx, cat)
            with tictoc('Saving matrix for %s to %s'%(cat, fn)):
                save_sparse_matrix(open(fn, 'wb'), arr)
        return

    def load_from_pfx(self, pfx):
        with tictoc('Loading TokenMaps'):
            self.ent_map, self.tm_list = pickle.load(open(pfx+'.pkl'))
        self.arr_list = []
        for cat in self.categories:
            self.arr_list.append(
                load_sparse_matrix(open('%s_%s.npz'%(pfx, cat), 'rb')))
        return


def print_proc_info():
    pid = os.getpid()
    print(pid, file=sys.stderr)



def print_config(msg=None, numpy=0, hostname=1, ps=1):
    import numpy
    import socket
    import os
    import psutil
    import sys
    try:
        if numpy:
            numpy.show_config()
        if msg is not None:
            print msg
        if hostname:
            print 'hostname', socket.gethostname()
        pid = os.getpid()
        print >> sys.stderr, 'pid', pid
        proc_stat = dict(e.split(':') for e in open('/proc/%d/status'%pid))
        print >> sys.stderr, 'VmHWM', proc_stat["VmHWM"] \
            'VmRSS', proc_stat["VmRSS"], \
            'VmSwap', proc_stat["VmSwap"], \
            'Threads', proc_stat["Threads"]
        p = psutil.Process(pid)
        print >> sys.stderr, 'CPU%        ',  '%1.2f%%'%p.cpu_percent(interval=None)
        print >> sys.stderr, 'MEM%        ', '%1.2f%%'%p.memory_percent()
    except Exception as e:
        print >> sys.stderr, e
    return
