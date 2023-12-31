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
    def __init__(
        self,
        name: str,
        expr: pl.expr.expr.Expr = None,
        expr_no_alias: pl.expr.expr.Expr = None,
        on_df=None,
        op=None,
    ):
        self._name = name
        self._expr = pl.col(name) if expr is None else expr
        self._expr_no_alias = (
            pl.col(name) if expr_no_alias is None else expr_no_alias
        )
        self._on_df = on_df
        self._op = op

    def asc(self):
        return self.asc_nulls_first()

    def desc(self):
        return self.desc_nulls_last()

    def asc_nulls_first(self):
        name = self._name + " ASC NULLS FIRST"
        expr_no_alias = self._expr_no_alias.sort(
            descending=False, nulls_last=False
        )
        expr = self._expr.sort(descending=False, nulls_last=False).alias(name)
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, "asc_nulls_first"),
        )

    def desc_nulls_first(self):
        name = self._name + " DESC NULLS FIRST"
        expr = self._expr.sort(descending=True, nulls_last=False).alias(name)
        expr_no_alias = self._expr_no_alias.sort(
            descending=True, nulls_last=False
        )
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, "desc_nulls_first"),
        )

    def asc_nulls_last(self):
        name = self._name + " ASC NULLS LAST"
        expr = self._expr.sort(descending=False, nulls_last=True).alias(name)
        expr_no_alias = self._expr.sort(descending=False, nulls_last=True)
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, "asc_nulls_last"),
        )

    def desc_nulls_last(self):
        name = self._name + " DESC NULLS LAST"
        expr = self._expr.sort(descending=True, nulls_last=True).alias(name)
        expr_no_alias = self._expr.sort(descending=True, nulls_last=True)
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, "desc_nulls_last"),
        )

    def alias(self, name: str):
        return Column(
            name,
            self._expr.alias(name),
            expr_no_alias=self._expr_no_alias.alias(name),
            op=(self, name, "alias"),
        )

    def __add__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} + {other._name})"
            expr = (self._expr + other._expr).alias(name)
            expr_no_alias = self._expr_no_alias + other._expr_no_alias
        else:
            name = f"({self._name} + {other})"
            expr = (self._expr + other).alias(name)
            expr_no_alias = self._expr_no_alias + other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__add__"),
        )

    def __sub__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} - {other._name})"
            expr = (self._expr - other._expr).alias(name)
            expr_no_alias = self._expr_no_alias - other._expr_no_alias
        else:
            name = f"({self._name} - {other})"
            expr = (self._expr - other).alias(name)
            expr_no_alias = self._expr_no_alias - other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__sub__"),
        )

    def __mul__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} * {other._name})"
            expr = (self._expr * other._expr).alias(name)
            expr_no_alias = self._expr_no_alias * other._expr_no_alias
        else:
            name = f"({self._name} * {other})"
            expr = (self._expr * other).alias(name)
            expr_no_alias = self._expr_no_alias * other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__mul__"),
        )

    def __truediv__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} / {other._name})"
            expr = (self._expr / other._expr).alias(name)
            expr_no_alias = self._expr_no_alias / other._expr_no_alias
        else:
            name = f"({self._name} / {other})"
            expr = (self._expr / other).alias(name)
            expr_no_alias = self._expr_no_alias / other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__truediv__"),
        )

    def __mod__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} % {other._name})"
            expr = (self._expr % other._expr).alias(name)
            expr_no_alias = self._expr_no_alias % other._expr_no_alias
        else:
            name = f"({self._name} % {other})"
            expr = (self._expr % other).alias(name)
            expr_no_alias = self._expr_no_alias % other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__mod__"),
        )

    def __pow__(self, other):
        if isinstance(other, Column):
            name = f"POWER({self._name}, {other._name})"
            expr = (self._expr**other._expr).alias(name)
            expr_no_alias = self._expr_no_alias**other._expr_no_alias
        else:
            name = f"POWER({self._name}, {other})"
            expr = (self._expr**other).alias(name)
            expr_no_alias = self._expr_no_alias**other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__pow__"),
        )

    def __neg__(self):
        name = f"(- {self._name})"
        expr = (-self._expr).alias(name)
        expr_no_alias = -self._expr_no_alias
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, "__neg__"),
        )

    def __invert__(self):
        name = f"(NOT {self._name})"
        expr = (~self._expr).alias(name)
        expr_no_alias = ~self._expr_no_alias
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, "__invert__"),
        )

    # abs is not supported in spark
    # def __abs__(self):
    #     return Column(
    #         f"(ABS({self._name}))",
    #         (self._expr.abs()).alias(f"(ABS({self._name}))"),
    #     )

    def startswith(self, other):
        if isinstance(other, Column):
            name = f"startswith({self._name}, {other._name})"
            expr_no_alias = self._expr_no_alias.str.startswith(
                other._expr_no_alias
            )
        else:
            name = f"startswith({self._name}, {other})"
            expr_no_alias = self._expr_no_alias.str.starts_with(other)
        return Column(
            name,
            expr_no_alias.alias(name),
            expr_no_alias=expr_no_alias,
            op=(self, other, "startswith"),
        )

    def endswith(self, other):
        if isinstance(other, Column):
            name = f"endswith({self._name}, {other._name})"
            expr = self._expr.str.endswith(other._expr).alias(name)
            expr_no_alias = self._expr_no_alias.str.endswith(
                other._expr_no_alias
            )
        else:
            name = f"endswith({self._name}, {other})"
            expr = self._expr.str.ends_with(other).alias(name)
            expr_no_alias = self._expr_no_alias.str.ends_with(other)
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "endswith"),
        )

    def substr(self, start, length):
        name = f"substring({self._name}, {start}, {length})"
        expr = self._expr.str.slice(
            start - 1 if start != 0 else 0, length
        ).alias(name)
        expr_no_alias = self._expr_no_alias.str.slice(
            start - 1 if start != 0 else 0, length
        )
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, start, length, "substr"),
        )

    def contains(self, other):
        if isinstance(other, Column):
            name = f"contains({self._name}, {other._name})"
            expr = self._expr.str.contains(other._expr).alias(name)
            expr_no_alias = self._expr_no_alias.str.contains(
                other._expr_no_alias
            )
        else:
            name = f"contains({self._name}, {other})"
            expr = self._expr.str.contains(other).alias(name)
            expr_no_alias = self._expr_no_alias.str.contains(other)
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "contains"),
        )

    def astype(self, dtype: Union[types.DataType, str]):
        if isinstance(dtype, str):
            dtype = types._type_name_to_dtype[dtype]
        cname = f"CAST({self._name} AS {dtype.simpleString().upper()})"
        return Column(
            cname,
            expr=self._expr.cast(types.map_type(dtype)).alias(cname),
            expr_no_alias=self._expr_no_alias.cast(types.map_type(dtype)),
            op=(self, dtype, "astype"),
        )

    def cast(self, dtype: Union[types.DataType, str]):
        return self.astype(dtype)

    def between(self, lowerBound, upperBound):
        _lowerBound = (
            lowerBound._expr
            if isinstance(lowerBound, Column)
            else pl.lit(lowerBound)
        )
        _upperBound = (
            upperBound._expr
            if isinstance(upperBound, Column)
            else pl.lit(upperBound)
        )
        _lowerBound_name = (
            lowerBound._name
            if isinstance(lowerBound, Column)
            else str(lowerBound)
        )
        _upperBound_name = (
            upperBound._name
            if isinstance(upperBound, Column)
            else str(upperBound)
        )
        return Column(
            f"(({self._name} >= {_lowerBound_name}) AND ({self._name} <= {_upperBound_name}))",
            self._expr.is_between(_lowerBound, _upperBound).alias(
                f"(({self._name} >= {_lowerBound_name}) AND ({self._name} <= {_upperBound_name}))"
            ),
            expr_no_alias=self._expr_no_alias.is_between(
                _lowerBound, _upperBound
            ),
        )

    def __and__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} AND {other._name})"
            expr = (self._expr & other._expr).alias(name)
            expr_no_alias = self._expr_no_alias & other._expr_no_alias
        else:
            name = f"({self._name} AND {other})"
            expr = (self._expr & other).alias(name)
            expr_no_alias = self._expr_no_alias & other
        return Column(
            name,
            expr_no_alias.alias(name),
            expr_no_alias=expr_no_alias,
            op=(self, other, "__and__"),
        )

    def __or__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} OR {other._name})"
            expr = (self._expr | other._expr).alias(name)
            expr_no_alias = self._expr_no_alias | other._expr_no_alias
        else:
            name = f"({self._name} OR {other})"
            expr = (self._expr | other).alias(name)
            expr_no_alias = self._expr_no_alias | other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__or__"),
        )

    def __gt__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} > {other._name})"
            expr = (self._expr > other._expr).alias(name)
            expr_no_alias = self._expr_no_alias > other._expr_no_alias
        else:
            name = f"({self._name} > {other})"
            expr = (self._expr > other).alias(name)
            expr_no_alias = self._expr_no_alias > other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__gt__"),
        )

    def __ge__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} >= {other._name})"
            expr = (self._expr >= other._expr).alias(name)
            expr_no_alias = self._expr_no_alias >= other._expr_no_alias
        else:
            name = f"({self._name} >= {other})"
            expr = (self._expr >= other).alias(name)
            expr_no_alias = self._expr_no_alias >= other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__ge__"),
        )

    def __lt__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} < {other._name})"
            expr = (self._expr < other._expr).alias(name)
            expr_no_alias = self._expr_no_alias < other._expr_no_alias
        else:
            name = f"({self._name} < {other})"
            expr = (self._expr < other).alias(name)
            expr_no_alias = self._expr_no_alias < other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__lt__"),
        )

    def __le__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} <= {other._name})"
            expr = (self._expr <= other._expr).alias(name)
            expr_no_alias = self._expr_no_alias <= other._expr_no_alias
        else:
            name = f"({self._name} <= {other})"
            expr = (self._expr <= other).alias(name)
            expr_no_alias = self._expr_no_alias <= other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__le__"),
        )

    def __eq__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} == {other._name})"
            expr = self._expr == other._expr.alias(name)
            expr_no_alias = self._expr_no_alias == other._expr_no_alias
        else:
            name = f"({self._name} == {other})"
            expr = (self._expr == other).alias(name)
            expr_no_alias = self._expr_no_alias == other
        return Column(
            name, expr, expr_no_alias=expr_no_alias, op=(self, other, "__eq__")
        )

    def __ne__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} != {other._name})"
            expr = self._expr != other._expr.alias(name)
            expr_no_alias = self._expr_no_alias != other._expr_no_alias
        else:
            name = f"({self._name} != {other})"
            expr = (self._expr != other).alias(name)
            expr_no_alias = self._expr_no_alias != other
        return Column(
            name,
            expr,
            expr_no_alias=expr_no_alias,
            op=(self, other, "__ne__"),
        )

    def isNull(self):
        return Column(
            f"({self._name} IS NULL)",
            self._expr.is_null().alias(f"({self._name} IS NULL)"),
            expr_no_alias=self._expr_no_alias.is_null(),
            op=(self, "isNull"),
        )

    def isNotNull(self):
        return Column(
            f"({self._name} IS NOT NULL)",
            self._expr.is_not_null().alias(f"({self._name} IS NOT NULL)"),
            expr_no_alias=self._expr_no_alias.is_not_null(),
            op=(self, "isNotNull"),
        )
