import sys
from collections import namedtuple
from itertools import zip_longest


def is_broadcastable(*shapes):

    if len(shapes) < 2:
        raise ValueError('must provide at least two shapes')

    if len(shapes) > 2:
        broadcasted_head = broadcast(shapes[0], shapes[1])
        out = is_broadcastable(*(
            (broadcasted_head,) + shapes[2:]))
    else:
        out = all(
                (m == n) or (m == 1) or (n == 1)
                for m, n in zip(shapes[0][::-1], shapes[1][::-1]))
    return out


def broadcast(*shapes):

    if len(shapes) < 2:
        raise ValueError('must provide at least two shapes')

    if len(shapes) > 2:
        broadcasted_head = broadcast(shapes[0], shapes[1])
        out = broadcast([broadcasted_head] + shapes[2:])
    else:
        if not is_broadcastable(shapes[0], shapes[1]):
            raise ValueError('shapes are not broadcastable')
        out = list(
                max(m, n) for m, n in
                zip_longest(
                    shapes[0][::-1], shapes[1][::-1],
                    fillvalue=1))[::-1]
    return out


def dict_to_namedtuple(typename, dictionary):
    namedtuple_typename = namedtuple(
            typename, dictionary.keys())
    return namedtuple_typename(**dictionary)


def erint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)
