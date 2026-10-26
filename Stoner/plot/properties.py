"""Resolve callable property proxies against an accessor's live rendering targets."""

import numpy as np


class PlotProperty:
    """Describe a Matplotlib property without retaining a Figure or Axes.

    Args:
        name (str):
            Property suffix shared by Matplotlib's ``get_...`` and ``set_...`` methods.

    Notes:
        Resolve each operation against the accessor's recent live Axes first,
        then its Figure. Getter and setter availability are checked separately.
        Descriptors are created on demand rather than installed on the shared
        accessor class, so properties of one projection cannot leak to another.
    """

    def __init__(self, name):
        self.name = name

    def method(self, accessor, operation):
        """Return a bound getter or setter, preserving Axes precedence."""
        for target in (accessor.ax, accessor.figure):
            method = getattr(target, f"{operation}_{self.name}", None)
            if callable(method):
                return method
        raise AttributeError(f"No live Axes or Figure provides {operation}_{self.name}.")

    def __get__(self, instance, owner=None):
        """Bind a callable value proxy when either operation is available."""
        if instance is None:
            return self
        try:
            self.method(instance, "get")
        except AttributeError:
            self.method(instance, "set")
        return PlotPropertyValue(self, instance)

    def __set__(self, instance, value):
        """Forward assignment as a single positional setter argument."""
        if isinstance(value, PlotPropertyValue):
            value = value.get()
        self.method(instance, "set")(value)


class PlotPropertyValue:
    """Expose a live property's value and its full getter and setter signatures.

    Args:
        descriptor (PlotProperty):
            Property description, independent of any particular artist.
        accessor (DataPlotter):
            Accessor whose current rendering targets are used for each operation.

    Notes:
        Calling without arguments gets the value; arguments select the setter.
        Use get and set explicitly when the getter requires arguments or a setter
        takes none. Representation, equality, indexing, iteration, formatting
        and NumPy conversion read the current value. Use get for an ordinary
        Python value, including arithmetic and identity or type checks.
    """

    def __init__(self, descriptor, accessor):
        self._descriptor = descriptor
        self._accessor = accessor

    def get(self, *args, **kwargs):
        """Call the native getter with all supplied arguments and return its value."""
        return self._descriptor.method(self._accessor, "get")(*args, **kwargs)

    def set(self, *args, **kwargs):
        """Call the native setter with all supplied arguments and return its result."""
        return self._descriptor.method(self._accessor, "set")(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Get with no arguments, or set with positional or keyword arguments."""
        return self.set(*args, **kwargs) if args or kwargs else self.get()

    def __getattr__(self, name):
        """Delegate public value attributes, such as string methods, to the getter."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.get(), name)

    def __repr__(self):
        return repr(self.get())

    def __str__(self):
        return str(self.get())

    def __format__(self, spec):
        return format(self.get(), spec)

    def __eq__(self, other):
        return self.get() == (other.get() if isinstance(other, PlotPropertyValue) else other)

    def __ne__(self, other):
        return self.get() != (other.get() if isinstance(other, PlotPropertyValue) else other)

    def __bool__(self):
        return bool(self.get())

    def __len__(self):
        return len(self.get())

    def __iter__(self):
        return iter(self.get())

    def __getitem__(self, key):
        return self.get()[key]

    def __float__(self):
        return float(self.get())

    def __int__(self):
        return int(self.get())

    def __array__(self, dtype=None, copy=None):
        return np.array(self.get(), dtype=dtype, copy=copy)
