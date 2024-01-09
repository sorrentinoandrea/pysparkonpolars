from typing import Union
from pysparkpl.sql.column import Column as col


def sum(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"sum({c._build_display_name()})", op=(c, "sum"))


def count(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"count({c._build_display_name()})", op=(c, "count"))


def avg(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"avg({c._build_display_name()})", op=(c, "avg"))


def mean(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"avg({c._build_display_name()})", op=(c, "mean"))


def min(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"min({c._build_display_name()})", op=(c, "min"))


def max(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"max({c._build_display_name()})", op=(c, "max"))


def countDistinct(c: Union[str, col]) -> col:
    c = c if isinstance(c, col) else col(c)
    return col(f"count({c._build_display_name()})", op=(c, "countDistinct"))
