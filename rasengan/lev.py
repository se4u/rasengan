'''
| Filename    : edit_distance.py
| Description : Find the edit distance between two strings.
| Author      : Pushpendre Rastogi
| Created     : Fri Dec 11 22:10:53 2015 (-0500)
| Last-Updated: Sat Dec 12 01:23:19 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 35
'''
import collections

EditDistance = collections.namedtuple(
    'EditDistance', ['substitute', 'insert', 'delete'])

def memoize(f):
    def g(*args):
        if args not in memoize.d:
            v = f(*args)
            memoize.d[args] = v
        else:
            v = memoize.d[args]
        return v
    return g

@memoize
def _levenshtein(a, b):
    '''
    Params
    ------
    a, b : strings to compare.
    Returns
    -------
    A tuple of (substitution, insertion, deletion) needed to convert
    a:str to b:str
    '''
    if not a:
        return tuple([0, len(b), 0])
    if not b:
        return tuple([0, 0, len(a)])
    sub = list(_levenshtein(a[1:], b[1:]))
    sub[0] += (a[0] != b[0])
    sub_sum = sum(sub)
    ins = list(_levenshtein(a, b[1:]))
    ins[1] += 1
    ins_sum = sum(ins)
    del_ = list(_levenshtein(a[1:], b))
    del_[2] += 1
    del_sum = sum(del_)
    if sub_sum < ins_sum:
        if sub_sum < del_sum:
            return tuple(sub)
        else:
            return tuple(del_)
    else:
        if ins_sum < del_sum:
            return tuple(ins)
        else:
            return tuple(del_)

def lev(a, b):
    memoize.d = {}
    v = EditDistance(*_levenshtein(a, b))
    del memoize.d
    return v

if __name__ == '__main__':
    assert lev('arg', 'arb') == EditDistance(1, 0, 0)
    assert lev('arg', 'brbc') == EditDistance(2, 1, 0)
    assert lev('argzzz', 'brbc') == EditDistance(3, 0, 2)
