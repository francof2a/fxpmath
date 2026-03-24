"""fxpmath

---

A python library for fractional fixed-point arithmetic.

---

This software is provided under MIT License:

MIT License

Copyright (c) 2020 Franco, francof2a

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

class Callback():
    """Base callback interface for `Fxp` events.
    
    Subclass this class and override one or more hook methods."""

    def __init__(self):
        """Initialize callback instance state.
        
        Side Effects
        ---
        Base implementation does not allocate state; subclasses can initialize
        custom fields here."""
        pass

    def on_value_change(self, fxp_object, logs=None):
        """Hook called after an `Fxp` value is updated.
        
        Parameters
        ---
        fxp_object : fxpmath.objects.Fxp
            Fixed-point object whose stored value was updated.
        logs : dict or None, optional
            Optional event payload provided by the caller.
        
        Side Effects
        ---
        Callback hook with no default side effect; subclasses can override to react to events."""
        pass

    def on_status_overflow(self, fxp_object, logs=None):
        """Hook called when overflow status is raised.
        
        Parameters
        ---
        fxp_object : fxpmath.objects.Fxp
            Fixed-point object that raised overflow status.
        logs : dict or None, optional
            Optional event payload provided by the caller.
        
        Side Effects
        ---
        Callback hook with no default side effect; subclasses can override to react to events."""
        pass

    def on_status_underflow(self, fxp_object, logs=None):
        """Hook called when underflow status is raised.
        
        Parameters
        ---
        fxp_object : fxpmath.objects.Fxp
            Fixed-point object that raised underflow status.
        logs : dict or None, optional
            Optional event payload provided by the caller.
        
        Side Effects
        ---
        Callback hook with no default side effect; subclasses can override to react to events."""
        pass

    def on_status_inaccuracy(self, fxp_object, logs=None):
        """Hook called when inaccuracy status is raised.
        
        Parameters
        ---
        fxp_object : fxpmath.objects.Fxp
            Fixed-point object that raised inaccuracy status.
        logs : dict or None, optional
            Optional event payload provided by the caller.
        
        Side Effects
        ---
        Callback hook with no default side effect; subclasses can override to react to events."""
        pass
