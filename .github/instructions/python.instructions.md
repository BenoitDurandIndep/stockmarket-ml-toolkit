---
applyTo: "**/*.py"
---

# Python File Rules

## Imports
- Always include `from __future__ import annotations` as the first import in every module.
- Group imports: standard library → third-party → local, separated by blank lines.
- Prefer explicit imports over wildcard (`from module import *`).

## Type Hints
- Annotate all function parameters and return types.
- Use `Optional[X]` (or `X | None` with `from __future__ import annotations`) for nullable values.
- Use `Dict`, `List`, `Tuple` from `typing` for compatibility; prefer built-in generics (`dict`, `list`) only when targeting Python 3.10+.

## Docstrings
- Every public function and class must have a Google-style docstring.
- Private helpers (prefixed `_`) should have at least a one-line docstring.
- Example:
  ```python
  def my_func(x: int, y: float) -> str:
      """Short description.

      Args:
          x (int): Description of x.
          y (float): Description of y.

      Returns:
          str: Description of return value.

      Raises:
          ValueError: If x is negative.
      """
  ```

## pandas / numpy
- Avoid row-wise iteration with `iterrows()`; use vectorised operations or `apply` with explicit `axis`.
- Always specify `axis` explicitly in `DataFrame.apply`, `DataFrame.sum`, etc.
- Use `pd.Series` boolean masks for filtering, not list comprehensions over DataFrame rows.

## Logging
- Use the `logging` module; never use `print` in library/module code.
- Obtain loggers with `logging.getLogger(__name__)`.

## Constants
- Define module-level constants in `UPPER_CASE` at the top of the file, below imports.

## Formatting
- Line length: 100 characters maximum.
- Use 4-space indentation (no tabs).
