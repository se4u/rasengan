'''
| Filename    : __init__.py
| Description : Handy decorators and context managers for improved REPL experience.
| Author      : Pushpendre Rastogi
| Created     : Thu Oct 29 19:43:24 2015 (-0400)
| Last-Updated: Thu Oct 29 21:07:11 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 15
'''

import contextlib
import time
import numpy
import random


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
    print "Started", msg
    yield
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
    numpy.random.seed(seed)
    random.seed(seed)
    yield
    if reset:
        numpy.random.seed(seed)
        random.seed(seed)


@contextlib.contextmanager
def debug_support(capture_ctrl_c=True):
    ''' Drop into
    '''
    try:
        import ipdb as pdb
    except ImportError:
        import pdb
    import traceback
    import sys
    import signal
    call_pdb = lambda _sig, _frame: pdb.set_trace()
    signal.signal(signal.SIGUSR1, call_pdb)
    if capture_ctrl_c:
        signal.signal(signal.SIGINT, call_pdb)
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
