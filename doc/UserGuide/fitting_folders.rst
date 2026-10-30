Fit collections of measurements
*******************************

Use :py:attr:`~Stoner.folders.each.Item.fit` to fit each measurement in a folder.
It provides the same algorithm choices and registered model names as
:doc:`fitting_registry`, and returns one result per direct member in folder order::

    import numpy as np
    from Stoner import Data, DataFolder
    from Stoner.enums import FitOutput

    folder = DataFolder(readlist=False)
    x = np.linspace(0, 3, 21)
    for index in range(3):
        folder += Data(x, (index + 2)*x + index - 1,
                       column_headers=["Field", "Signal"], setas="xy")

    reports = folder.each.fit.linear(
        x="Field", y="Signal", output=FitOutput.REPORT,
        result=True, header="Fitted signal", _serial=True,
    )
    print([report.params["slope"].value for report in reports])

The slopes are approximately 2, 3 and 4. Each member retains its own fitted
parameters and fitted column. Omitted coordinates use that member's own roles.
Looking up the proxy, inspecting ``folder.each.fit.models`` or completing a model
name does not load measurements. Invoking a fit retains normal bulk prefetching.

Choose algorithms and collect results
======================================

The callable form defaults to lmfit. Explicit algorithms accept the same fitting
arguments as their ``Data.fit`` counterparts::

    from Stoner.analysis.fitting.models.generic import Linear

    rows = folder.each.fit(Linear, output=FitOutput.ROW, _serial=True)
    rows = folder.each.fit.lmfit(Linear, output=FitOutput.ROW, _serial=True)
    rows = folder.each.fit.odr(Linear, output=FitOutput.ROW, _serial=True)
    fits = folder.each.fit.curve_fit(Linear, _serial=True)
    coefficients = folder.each.fit.polyfit(polynomial_order=1, _serial=True)

``differential_evolution`` is also available. Coordinates use ``x`` and ``y``;
legacy ``xcol``/``ycol`` keywords are rejected by the member accessor. Existing
direct bulk methods such as ``folder.each.lmfit`` remain available during migration.

All output modes retain their existing member-level contents. For example,
``FitOutput.DATA`` returns the updated measurements; the normal bulk replacement
rules retain them in the folder. Other outputs include reports, arrays or tuples.
An empty folder returns an empty list.

The existing ``_return`` control optionally stores each returned value in its
member's metadata::

    rows = folder.each.fit.linear(output=FitOutput.ROW, _return="fit_summary")
    rows = folder.each.fit.linear(output=FitOutput.ROW, _return=True)

The second form uses the final method name as the metadata key: ``linear`` for
the shortcut, ``lmfit`` for the explicit algorithm, and ``fit`` for a default
call. The standard bulk ``args``, ``kwargs``, ``func`` and ``setas`` tracking is
retained. Positional arguments remain in ``args`` rather than being duplicated
as keyword arguments. These controls are consumed by the bulk layer.

String dispatch and streaming are supported::

    rows = folder.each("fit.linear", output=FitOutput.ROW, _serial=True)
    for row in folder.each.iter("fit.linear", output=FitOutput.ROW, _serial=True):
        print(row)

Worker failures retain the standard ``(exception, traceback)`` result tuple;
later members continue. An ordinary fitted-parameters result can also be a tuple,
so check whether its first element is an exception when inspecting failures.

Serial, thread and process execution
====================================

Fitting follows the existing bulk pool-selection policy. ``_serial=True`` forces
serial execution; otherwise the ``Options.multiprocessing`` and
``Options.threading`` settings select the existing pool backend. Plotting keeps
its separate serial policy. For a process-based script, guard its entry point::

    from Stoner import Options

    if __name__ == "__main__":
        Options.multiprocessing = True
        Options.threading = False
        reports = folder.each.fit.linear(output=FitOutput.REPORT)

Keep worker-count and pool-creation behaviour consistent with the existing
folder options. Member results remain in folder order for every backend.

Custom registered shortcuts are resolved at the start of each bulk invocation.
The selected registration is sent with the task, and each worker obtains its
own member-bound accessor and model state. This includes explicit replacement
of a built-in registry key. A shortcut saved before replacement uses the current
registration when the bulk call starts. The worker's global registry is unchanged;
a custom factory should not assume its own name is installed there.

The existing serializer supports ordinary built-in models and tested local
``@make_model`` classes/factories. A definition capturing unsupported external
resources can still fail transfer. A failed fitting-call preflight raises a
``TypeError`` identifying the call; use serial execution or supply a serialisable
definition. It does not silently change backend to handle a serialisation failure.
Per-member fitting errors continue to use the result tuples described above.

The operation closes and joins its own pool after completion. Transfer failure
or closing a partially consumed iterator terminates and joins that operation's
pool. It does not terminate unrelated pools.

Existing instrument-data example
================================

This runnable example loads the retained non-local IV fixtures, fits each
measurement, and collects their fitted metadata. It also demonstrates an explicit
bulk ODR call:

.. literalinclude:: /samples/folder_fit.py
   :language: python

See :doc:`datafolder` for loading and grouping collections and
:doc:`plot_folders` for displaying the results.
