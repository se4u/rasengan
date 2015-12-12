'''
| Filename    : __init__.py
| Description : Handy decorators and context managers for improved REPL experience.
| Author      : Pushpendre Rastogi
| Created     : Thu Oct 29 19:43:24 2015 (-0400)
| Last-Updated: Wed Dec  9 16:24:22 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 113
'''
import collections
import contextlib
import time
import numpy
import random
import print_hook
import sys


def print_indent_fn(text):
    if len(text) > 0:
        indent = print_indent_fn.indent
        white_space = (2 * indent * ' ')
        return text.replace('\n', '\n' + white_space)
    else:
        return text


def setup_print_indent():
    print_indent_fn.indent = 0
    setup_print_indent.printhook = print_hook.PrintHook().start(
        func=print_indent_fn, override='stdout')
    return setup_print_indent.printhook


def increase_print_indent():
    try:
        print_indent_fn.indent += 1
    except AttributeError:
        setup_print_indent()
        increase_print_indent()


def decrease_print_indent():
    print_indent_fn.indent -= 1


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
    t = time.time()
    increase_print_indent()
    print "Started", msg
    yield
    decrease_print_indent()
    print
    print "Completed", msg, "in %0.1fs" % (time.time() - t)


class announce(object):

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


class reseed(object):

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


@contextlib.contextmanager
def debug_support(capture_ctrl_c=True):
    ''' Drop into
    '''
    try:
        import ipdb as pdb
    except ImportError:
        import pdb
    import traceback
    import signal
    import code

    def top_frame(frame):
        if frame.f_back is None:
            return frame
        else:
            return top_frame(frame.f_back)

    call_pdb = (lambda _sig, _frame:
                code.InteractiveConsole(dict(_frame=_frame, top_frame=top_frame)).interact(
                    "Entering python shell. Press Ctrl-d to resume execution."))
    signal.signal(signal.SIGUSR1, call_pdb)
    if capture_ctrl_c:
        signal.signal(signal.SIGINT, call_pdb)
        pass
    try:
        yield
    except:
        traceback.print_exc()
        pdb.post_mortem(sys.exc_info()[2])

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
        if isinstance(e, list):
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

    def __getitem__(self, key):
        return self.obj.__getitem__(key)

    def __setitem__(self, key, value):
        self.obj.__setitem__(key, value)

    def __delitem__(self, key):
        self.obj.__delitem__(key)

    def __iter__(self):
        return self.obj.__iter__()

    def __len__(self):
        return self.obj.__len__()

    def __add__(self, right_obj):
        return (self.obj.__add__(right_obj.obj)
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
        print ' '.join(tknzr.tokenize(row))


def pipeline_dictionary(tokenize=0, lowercase=0):
    ''' This function is called from the commandline to extract a dictionary
    from a file after tokenizing it.
    '''
    tokenizer = (get_tokenizer()
                 if tokenize
                 else
                 (lambda x: x.split(' ')))
    d = collections.defaultdict(int)
    for row in sys.stdin:
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
