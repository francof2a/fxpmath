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

## migration notes for NumPy `1.26.x` to `2.x`

If you are upgrading an existing project from NumPy `1.26.x` to `2.x`, these are the main points to check:

* Review any call sites that rely on `np.array(..., copy=False)` or similar strict no-copy assumptions. NumPy `2.x` is stricter and may raise where `1.26.x` returned a converted array.
* Prefer direct `fxpmath` APIs or `Fxp` object methods when you need `fxpmath`-specific output controls such as `out_like`. Top-level NumPy functions may reject those keywords before dispatch.
* Keep explicit `Fxp` outputs when using fxpmath-managed NumPy dispatch with `out`. Passing non-`Fxp` outputs is expected to fail with `TypeError`.
* Re-run your fixed-point NumPy integration tests on both supported version tracks if your project depends on custom wrappers, uncommon keyword arguments, or subclass-sensitive array conversion behavior.

A practical upgrade workflow is:

1. Pin your project first to NumPy `1.26.4` and make sure your `fxpmath` tests are green.
2. Upgrade to the target NumPy `2.x` release and rerun the same suite.
3. Check failures first around array conversion, `out` handling, and unsupported keyword arguments.
4. Replace top-level NumPy call sites with direct `fxpmath` APIs where you need fixed-point-specific behavior.

## complex wrapping and casting behavior

Complex values in wrap/overflow paths are handled component-wise (real and imaginary parts independently).

* Supported complex wrap paths should not emit NumPy `ComplexWarning`.
* Real/imaginary components are wrapped using the same signed/word-size policy as non-complex values.

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
