run.spec selection grammar
===========================


``garak/_spec.py`` implements the unified ``run.spec`` selection grammar: a
single internal ``Spec`` (a list of ``Selector`` with explicit polarity) that
both transports parse to.

* CLI string (comma separated), via ``parse_spec_string``
* config file form (YAML/JSON ``include``/``exclude`` lists), via ``parse_spec_file``

Selectors carry a category-prefixed plugin path (``probes.<module>[.<Class>]``,
``buffs.<module>[.<Class>]``) or a probe filter (``tag:<prefix>``,
``tier:<N|name>``). A leading ``-`` excludes; ``tier:N`` is inclusive
("log level": tiers ``1..N``). ``Spec.resolve()`` orchestrates probe and buff
selection plus the filters; the single plugin-path resolution core,
``_resolve_plugin_paths``, is shared with the ``parse_plugin_spec`` adapter used
for detectors.

See :doc:`configurable` for the user-facing grammar and examples.


Code
^^^^


garak._spec
-----------

.. automodule:: garak._spec
   :members:
   :undoc-members:
   :show-inheritance:
