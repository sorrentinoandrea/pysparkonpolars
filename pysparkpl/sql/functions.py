from typing import Union
from pysparkpl.sql.column import Column as col


def sum(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"sum({c._name})", c._expr.sum().alias(f"sum({c._name})"))


def count(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"count({c._name})", c._expr.count().alias(f"count({c._name})"))


def avg(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"avg({c._name})", c._expr.mean().alias(f"avg({c._name})"))


def mean(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"avg({c._name})", c._expr.mean().alias(f"avg({c._name})"))


def min(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"min({c._name})", c._expr.min().alias(f"min({c._name})"))


def max(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"max({c._name})", c._expr.max().alias(f"max({c._name})"))


def countDistinct(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(
        f"count({c._name})",
        c._expr.n_unique().alias(f"count({c._name})"),
    )
