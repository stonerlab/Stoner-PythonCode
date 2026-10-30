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

   .. rubric:: Fitting accessor

   .. autoattribute:: fit

   .. rubric:: Plotting accessor

   .. autoattribute:: plot
{% endif %}
{% if fullname == 'Stoner.Image.core.ImageFile' %}

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
