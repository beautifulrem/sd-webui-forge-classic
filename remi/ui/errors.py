"""Validation errors of the parameter model."""

from __future__ import annotations


class SchemaError(ValueError):
    """A layout tree that cannot be exported or bound (duplicate names, bad
    conditions, ...): a programming error of the script that declared it."""


class ParamError(ValueError):
    """One rejected value. ``owner`` and ``param`` are filled in by the layout
    that validated it."""

    def __init__(self, message: str, *, param: str | None = None, owner: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.owner = owner

    def to_json(self) -> dict:
        return {"owner": self.owner, "param": self.param, "message": self.message}

    def __str__(self) -> str:
        where = ".".join(part for part in (self.owner, self.param) if part)
        return f"{where}: {self.message}" if where else self.message


class ValidationError(ValueError):
    """Every rejected value of a request, as ``{"errors": [...]}``."""

    def __init__(self, errors: list[ParamError]):
        self.errors = list(errors)
        super().__init__("; ".join(str(error) for error in self.errors) or "invalid values")

    def to_json(self) -> dict:
        return {"errors": [error.to_json() for error in self.errors]}

    @classmethod
    def merge(cls, errors: list[ValidationError | ParamError]) -> ValidationError:
        flat: list[ParamError] = []
        for error in errors:
            flat.extend(error.errors if isinstance(error, ValidationError) else [error])
        return cls(flat)
