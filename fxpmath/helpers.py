"""
Internal helpers for fixed-point operation kernels.
"""

import numpy as np

from . import _n_word_max


def _cast_to_object(x):
    return np.array(x, dtype=object)


def _cast_func(use_object):
    return _cast_to_object if use_object else (lambda m: m)


def _requires_object_for_scale(n_word, shift):
    shift = int(shift)

    if n_word is None:
        return False

    n_word = int(n_word)

    if n_word >= _n_word_max:
        return True

    if shift >= (_n_word_max - 1):
        return True

    if shift > 0 and (n_word + shift) >= _n_word_max:
        return True

    return False


def _use_object_cast(scale_terms=None, product_terms=None, pow2_terms=None):
    if scale_terms is not None:
        for n_word, shift in scale_terms:
            if _requires_object_for_scale(n_word, shift):
                return True

    if product_terms is not None:
        for terms in product_terms:
            if isinstance(terms, (tuple, list)):
                total_bits = int(np.sum(terms))
            else:
                total_bits = int(terms)

            if total_bits >= _n_word_max:
                return True

    if pow2_terms is not None:
        for shift in pow2_terms:
            if int(shift) >= (_n_word_max - 1):
                return True

    return False
