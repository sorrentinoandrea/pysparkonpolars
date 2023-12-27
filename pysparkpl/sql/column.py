from typing import Union
import polars as pl
import pysparkpl.sql.types as types

#  'alias',
#  'asc',
#  'asc_nulls_first',
#  'asc_nulls_last',
#  'astype',
#  'between',
#  'bitwiseAND',
#  'bitwiseOR',
#  'bitwiseXOR',
#  'cast',
#  'contains',
#  'desc',
#  'desc_nulls_first',
#  'desc_nulls_last',
#  'dropFields',
#  'endswith',
#  'eqNullSafe',
#  'getField',
#  'getItem',
#  'ilike',
#  'isNotNull',
#  'isNull',
#  'isin',
#  'like',
#  'name',
#  'otherwise',
#  'over',
#  'rlike',
#  'startswith',
#  'substr',
#  'when',
#  'withField']


class Column:
    def __init__(self, name: str, expr: pl.expr.expr.Expr = None):
        self._name = name
        self._expr = pl.col(name) if expr is None else expr

    def asc(self):
        return self.asc_nulls_first()

    def desc(self):
        return self.desc_nulls_last()

    def asc_nulls_first(self):
        return Column(
            self._name + " ASC NULLS FIRST",
            self._expr.sort(descending=False, nulls_last=False).alias(
                self._name
            ),
        )

    def desc_nulls_first(self):
        return Column(
            self._name + " DESC NULLS FISRT",
            self._expr.sort(descending=True, nulls_last=False).alias(
                self._name
            ),
        )

    def asc_nulls_last(self):
        return Column(
            self._name + " ASC NULLS LAST",
            self._expr.sort(descending=False, nulls_last=True).alias(
                self._name
            ),
        )

    def desc_nulls_last(self):
        return Column(
            self._name + " DESC NULLS LAST",
            self._expr.sort(descending=True, nulls_last=True).alias(self._name),
        )

    def alias(self, name: str):
        return Column(name, self._expr.alias(name))

    def __add__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} + {other._name})",
                (self._expr + other._expr).alias(
                    f"({self._name} + {other._name})"
                ),
            )
        return Column(
            f"({self._name} + {other})",
            (self._expr + other).alias(f"({self._name} + {other})"),
        )

    def __sub__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} - {other._name})",
                (self._expr - other._expr).alias(
                    f"({self._name} - {other._name})"
                ),
            )
        return Column(
            f"({self._name} - {other})",
            (self._expr - other).alias(f"({self._name} - {other})"),
        )

    def __mul__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} * {other._name})",
                (self._expr * other._expr).alias(
                    f"({self._name} * {other._name})"
                ),
            )
        return Column(
            f"({self._name} * {other})",
            (self._expr * other).alias(f"({self._name} * {other})"),
        )

    def __truediv__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} / {other._name})",
                (self._expr / other._expr).alias(
                    f"({self._name} / {other._name})"
                ),
            )
        return Column(
            f"({self._name} / {other})",
            (self._expr / other).alias(f"({self._name} / {other})"),
        )

    def __mod__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} % {other._name})",
                (self._expr % other._expr).alias(
                    f"({self._name} % {other._name})"
                ),
            )
        return Column(
            f"({self._name} % {other})",
            (self._expr % other).alias(f"({self._name} % {other})"),
        )

    def astype(self, dtype: Union[types.DataType, str]):
        if isinstance(dtype, str):
            dtype = types._type_name_to_dtype[dtype]
        cname = f"CAST({self._name} AS {dtype.simpleString().upper()})"
        return Column(
            cname,
            self._expr.cast(types.map_type(dtype)).alias(cname),
        )

    def __and__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} AND {other._name})",
                (self._expr & other._expr).alias(
                    f"({self._name} AND {other._name})"
                ),
            )
        return Column(
            f"({self._name} AND {other})",
            (self._expr & other).alias(f"({self._name} AND {other})"),
        )

    def __or__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} OR {other._name})",
                (self._expr | other._expr).alias(
                    f"({self._name} OR {other._name})"
                ),
            )
        return Column(
            f"({self._name} OR {other})",
            (self._expr | other).alias(f"({self._name} OR {other})"),
        )

    def __gt__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} > {other._name})",
                (self._expr > other._expr).alias(
                    f"({self._name} > {other._name})"
                ),
            )
        return Column(
            f"({self._name} > {other})",
            (self._expr > other).alias(f"({self._name} > {other})"),
        )

    def __ge__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} >= {other._name})",
                (self._expr >= other._expr).alias(
                    f"({self._name} >= {other._name})"
                ),
            )
        return Column(
            f"({self._name} >= {other})",
            (self._expr >= other).alias(f"({self._name} >= {other})"),
        )

    def __lt__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} < {other._name})",
                (self._expr < other._expr).alias(
                    f"({self._name} < {other._name})"
                ),
            )
        return Column(
            f"({self._name} < {other})",
            (self._expr < other).alias(f"({self._name} < {other})"),
        )

    def __le__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} <= {other._name})",
                (self._expr <= other._expr).alias(
                    f"({self._name} <= {other._name})"
                ),
            )
        return Column(
            f"({self._name} <= {other})",
            (self._expr <= other).alias(f"({self._name} <= {other})"),
        )

    def __eq__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} == {other._name})",
                (self._expr == other._expr).alias(
                    f"({self._name} == {other._name})"
                ),
            )
        return Column(
            f"({self._name} == {other})",
            (self._expr == other).alias(f"({self._name} == {other})"),
        )

    def __ne__(self, other):
        if isinstance(other, Column):
            return Column(
                f"({self._name} != {other._name})",
                (self._expr != other._expr).alias(
                    f"({self._name} != {other._name})"
                ),
            )
        return Column(
            f"({self._name} != {other})",
            (self._expr != other).alias(f"({self._name} != {other})"),
        )
