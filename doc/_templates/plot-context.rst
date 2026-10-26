{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

Available directly as ``data.plot.{{ objname }}`` on a
:py:class:`~Stoner.core.data.Data` object. This is a direct stonerplots export;
the signature and documentation below come from the installed dependency.

.. autoclass:: {{ objname }}
{% if objname == 'TexEngFormatter' %}
   :class-doc-from: both
{% endif %}

See :doc:`../plot-contexts` for the complete context and formatter list and
:doc:`../UserGuide/plot_layouts` for worked examples through ``data.plot``.
