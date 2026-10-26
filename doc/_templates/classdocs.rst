{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
    :toctree:
   {% for item in methods %}
      ~{{ objname }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}
{% if fullname == 'Stoner.core.data.Data' %}

   .. rubric:: Plotting accessor

   .. autoattribute:: plot
{% endif %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ objname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
