# Docstring standard

This is the expected standard for new and revised Stoner docstrings. It is derived from the package's existing
conventions and resolves their inconsistent variants. Existing departures are not alternative standards; bring a
docstring into line when editing it, while preserving its technical meaning and checking it against the implementation.

## Language and style

- Write prose in British English, including spellings such as behaviour, colour, normalise, centre and initialise.
- Preserve the exact spelling and case of identifiers, keyword arguments, external API names, filenames, literal values
  and quotations. Do not rename an API to make its spelling British English.
- Use a short, capitalised summary ending with a full stop. For functions and methods, use the imperative form, such as
  "Return the selected column." or "Calculate the propagated uncertainty." Describe a class or module's purpose directly.
- Explain the operation and its contract rather than merely restating the function name or signature.
- Keep prose readable and wrap long lines within the repository's 120-character line-length convention.

## Overall format

Use Google-style sections with reStructuredText/Sphinx markup. Sphinx also parses NumPy-style sections in imported
third-party docstrings, but Stoner's authoring convention remains Google style. Do not use underlined NumPy sections
or Sphinx `:param:`/`:type:` field lists as an alternative parameter format.

- Use triple double quotes, with the summary on the same line as the opening quotes.
- Use a one-line docstring for a short helper or property when the summary fully describes its contract.
- For a multiline docstring, leave a blank line after the summary and between sections. Put the closing quotes on their
  own line, aligned with the opening quotes.
- Indent section contents by four spaces relative to the section heading. Indent an entry's description by a further
  four spaces, placing that description on the line after its name/type label.
- Use a raw docstring (`r"""..."""`) when backslashes in mathematical or other literal content must be preserved.
- Include only relevant sections; do not add empty sections or padding text.

## Section names and contents

Use these canonical headings, with a trailing colon. Use the order below when the sections are present; `Returns:` and
`Yields:` describe different contracts, so select the applicable one rather than duplicating the same information.

| Heading | Required content when applicable |
|---|---|
| `Args:` | Main input arguments, using their exact signature names. |
| `Keyword Arguments:` | Optional configuration arguments and supported forwarded keyword arguments. |
| `Returns:` | Return type and meaning, including conditional return behaviour. |
| `Yields:` | Type and meaning of each value yielded by an iterator or generator. |
| `Raises:` | Relevant exception types and the conditions under which callers encounter them. |
| `Attributes:` | Public attributes of a class, with their types and meanings. |
| `Notes:` | Assumptions, algorithms, scientific interpretation, limitations and side effects. |
| `See Also:` | Related APIs or user-guide sections, using resolvable cross-references. |
| `Example:` | A useful code example or an embedded documentation plot. |
| `References:` | Scientific or technical sources relevant to the implementation. |

Use `Args:`, `Keyword Arguments:`, `Notes:` and `Example:` consistently. Do not alternate with `Arguments:`,
`Keyword Args:`, `Note:` or `Examples:`. The section order above standardises the variable order in existing docstrings.

## Arguments and types

Write each argument or attribute as `name (type):`, followed by its indented description on the next line. Include the
type in the docstring even if the signature has an annotation. For returns and yields, use `type:` followed by the
indented description, without an extra pair of parentheses around the type. For exceptions, use `ExceptionType:`
followed by the condition that raises it.

- Describe defaults in the argument's prose, and explain the meaning of `None`, sentinel values and accepted alternatives.
- Explain domain-specific types such as column indices, array shapes, units and value/uncertainty pairs precisely.
  Avoid uninformative type descriptions such as "various".
- Keep argument names, types, defaults and described behaviour consistent with the actual signature and implementation.
- The split between `Args:` and `Keyword Arguments:` describes the API's main inputs and optional controls. It does not
  imply that the latter are keyword-only in Python. Document positional-only or keyword-only restrictions when relevant.
- Do not document an ordinary bound method's implicit `self` or `cls` as a caller-supplied argument.
- For functions dynamically attached to `Data`, document the explicit `datafile` argument. Explain that it supplies the
  instance when the function is called without being used as a bound method.
- Document supported `*args` and `**kwargs`, including where forwarded arguments are consumed. Do not imply that arbitrary
  keyword arguments are accepted when only a defined set is supported.

## Scientific and behavioural contracts

Document whether an operation modifies its input, returns the same object for chaining, returns a copy, or produces a
separate result. Explain any options that change that behaviour. Do not describe a returned object as a copy unless the
implementation actually makes one.

Where relevant, describe effects on metadata, masks, column headers, column roles (`setas`), filenames and uncertainties.
State array dimensions, units, mathematical assumptions, boundary handling and uncertainty propagation explicitly.
Extended explanations belong in `Notes:` rather than unlabelled prose following a return description.

Class docstrings must describe the class's purpose and public attributes. Document constructor arguments where the
constructor is documented, keeping that description consistent with the class overview. Property getter and setter
summaries must describe their respective operations; use multiline descriptions when validation or side effects need
explanation. Use Sphinx `#:` comments for descriptor or class-attribute documentation where appropriate.

## Sphinx markup and examples

- Use reStructuredText rather than Markdown inside docstrings. Use double backticks for inline code/literal values and
  `*parameter*` for an emphasised parameter name in prose.
- Use `:py:class:`, `:py:meth:`, `:py:func:` and `:py:attr:` for Python object references, and `:ref:` for labelled guide
  sections. Select the role that matches the target object.
- Verify that each cross-reference resolves to the current object. Use qualified names to avoid ambiguous or stale
  references; a leading `~` can shorten the displayed name while retaining the qualified target.
- Use `:math:` for inline mathematics and `.. math::` for displayed equations.
- Put examples under `Example:`. Use `>>>` examples or properly indented reStructuredText code blocks for code.
- For existing documentation-example integration, use `.. plot::` with the appropriate sample script and its options,
  following the plot-cache policy in `AGENTS.md`.
- Keep examples consistent with the supported public API and the behaviour described by the docstring.

## Template

The following illustrates the format; adapt the content to the actual function rather than copying its behavioural
claims unchanged.

```python
"""Transform a selected data column.

Args:
    datafile (Data):
        Data object to work with if not used as a bound method.
    col (int or str):
        Index or header identifying the column.

Keyword Arguments:
    replace (bool):
        Replace the original column. Defaults to False.

Returns:
    Data:
        The modified object, allowing chained operations.

Notes:
    Explain the operation's assumptions, metadata effects and uncertainty handling.
"""
```

## Basis and enforcement

The source survey covered 1,443 docstrings in 87 package files. Representative examples include
[SG_Filter](Stoner/analysis/filtering.py), [rolling_window](Stoner/core/methods.py), the
[Data class](Stoner/core/data.py), [property and descriptor documentation](Stoner/core/property.py), and
[stitch](Stoner/analysis/functions.py). Parser settings are in [doc/conf.py](doc/conf.py).

The existing lint configurations are not a consistent formal specification of this style. This document is the writing
standard; reconcile tooling with it when updating lint configuration. Typographical errors, stale names, missing
arguments, inaccurate return descriptions and malformed markup in older docstrings are defects to correct, not
conventions to reproduce.
