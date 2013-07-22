"""Convenience functions for benchmarking."""

import time
import os
import os.path

import numpy as np


try:
    import builtins
except ImportError:  # Python 3x
    import __builtin__ as builtins

cache_tools_available = False
try:
    from cache_tools import dontneed
except ImportError:
    pass
else:
    cache_tools_available = True


# profile() is defined by most profilers, these lines defines it even if
# there is no active profiler.
class FakeProfile(object):
    def __call__(self, func):
        return func

    def timestamp(self, msg=None):  # defined by memory profiler
        class _FakeTimeStamper(object):
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        return _FakeTimeStamper()

if 'profile' not in builtins.__dict__:
    builtins.__dict__["profile"] = FakeProfile()


# A crude timer
def timeit(f):
    """Decorator for function execution timing."""
    def timed(*arg, **kwargs):
        if hasattr(f, "func_name"):
            fname = f.func_name
        else:
            fname = "<unknown>"
        print("Running %s() ..." % fname)
        start = time.time()
        ret = f(*arg, **kwargs)
        end = time.time()
        print("Elapsed time for %s(): %.3f s"
              % (fname, (end - start)))
        return ret
    return timed


def cache_array(value, filename, decimal=7):
    """Helper function for checking that a value hasn't changed between
    two invocations.

    First call: write value is a file
    Second call: check that what was written is identical to the value
        provided in the second call.
        TODO: only numpy arrays are compared, other values still have to
        be compared.

    Parameters
    ==========
    value: arbitrary Python value
        this could include numpy objects. Uses persistence from joblib to
        achieve high efficiency.

    """
    from joblib.numpy_pickle import dump, load
    base_dir = os.path.split(filename)[0]
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    if os.path.isfile(filename):
        cached = load(filename)
        np.testing.assert_almost_equal(cached, value, decimal=decimal)
    else:
        dump(value, filename)
