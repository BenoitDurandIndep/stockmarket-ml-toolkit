# General Coding Rules

## Naming
- Use `snake_case` for variables, functions, and module names.
- Use `PascalCase` for classes.
- Use `UPPER_CASE` for module-level constants (e.g. `RANDOM_KEY`, `BUY_ORDER`).
- Prefix private helpers with a single underscore (e.g. `_require_models`).

## Functions & Methods
- Keep functions small and focused on a single responsibility.
- Prefer explicit return types over implicit `None` returns.
- Validate inputs early and raise descriptive exceptions (`ValueError`, `TypeError`).
- Avoid mutable default arguments; use `None` and assign inside the function body.

## Error Handling
- Raise specific built-in exceptions with a clear message.
- Never silently swallow exceptions with a bare `except:`.
- Log errors at the appropriate level before re-raising when relevant.

## Comments & Documentation
- Write docstrings for every public function, method, and class.
- Use Google-style docstrings with `Args`, `Returns`, and `Raises` sections.
- Inline comments should explain *why*, not *what*.

## General
- Prefer readability over cleverness.
- Avoid magic numbers; assign them to named constants.
- Do not leave dead code or commented-out blocks in committed files.
- Each file should have a one-line module docstring at the top.
