# compatibility notes

This page collects the main compatibility details that users should know when working with `fxpmath` `0.5.0-dev`.

## supported versions

Current compatibility targets are:

* Python `3.9` to `3.14`
* NumPy `1.26.x` and `2.x`

Package metadata currently declares:

```text
numpy>=1.26.4,<3
```

## NumPy dispatch behavior

`fxpmath` supports NumPy interoperability through `__array_ufunc__` and `__array_function__`, but not every NumPy call path exposes the same keyword and output behavior.

### `out` behavior

For fxpmath-managed NumPy dispatch paths:

* `out` must be an `Fxp` object when it is provided explicitly.
* `out=...` is treated the same as omitting `out`.
* Passing a non-`Fxp` `out` raises `TypeError`.

This keeps result typing explicit and avoids ambiguous output casting.

### `out_like` behavior

`out_like` is an `fxpmath` extension, not a standard NumPy keyword.

That means:

* `out_like` is supported by direct `fxpmath` APIs such as `fxpmath.functions.*`.
* `out_like` is also supported by `Fxp` object methods that forward into fxpmath-managed operations.
* Top-level NumPy functions may reject `out_like` before dispatch, because NumPy treats it as an unknown keyword argument.

If you need template-based output control, prefer direct fxpmath APIs or object methods instead of relying on top-level NumPy calls.

## `np.array(..., copy=False)` differences

Array conversion behavior differs between NumPy `1.26.x` and `2.x`.

* On NumPy `2.x`, `np.array(x, dtype=..., copy=False)` can raise when a copy would be required.
* On NumPy `1.26.x`, the same call may still return a converted array.

If your code depends on strict no-copy semantics, test it on the NumPy version range you plan to support.

## unsupported or unknown kwargs

Unsupported keyword arguments should fail explicitly.

This is important for two reasons:

* some NumPy entry points reject unknown keywords before fxpmath dispatch runs
* fxpmath should not silently reinterpret unsupported keywords in ways that hide bugs

If a keyword is not documented by NumPy for that function and is not documented by fxpmath for that API, expect a `TypeError`.

## backward compatibility guidance

This development line keeps backward compatibility as a priority, but users should still be aware of a few practical points:

* public imports remain stable even after internal module reorganization
* NumPy interoperability is supported across both `1.26.x` and `2.x`, but edge behavior can differ around dispatch and copy semantics
* direct fxpmath APIs remain the most reliable way to access fxpmath-specific extensions such as `out_like`

When writing reusable downstream code, prefer explicit fxpmath call sites whenever you need fixed-point-specific output control.
