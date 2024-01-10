from typing import Union, Tuple, Set
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

_SEPARATOR = "__#_#_#__"


def _joined_col_name(col_name: str, from_df=None):
    if from_df is None:
        return col_name
    name = col_name
    if "__#_#_#__" not in name:
        name += "__#_#_#__"
    return f"{name}__on_df__{hex(id(from_df))}"


def _extract_name_and_dfs_from_col_name(col_name: str) -> Tuple[str, Set[str]]:
    if "__#_#_#__" not in col_name:
        return col_name, set()
    name, from_dfs = col_name.split("__#_#_#__")
    from_dfs = set(from_dfs.split("__on_df__")).difference({""})
    return name, from_dfs


def _identify_column(
    col_name: str, on_df=None, raise_if_not_found=True, raise_if_ambigous=True
):
    name, from_dfs = _extract_name_and_dfs_from_col_name(col_name)
    cols_info = {
        c: _extract_name_and_dfs_from_col_name(c) for c in on_df._df.columns
    }
    if on_df is None or len(from_dfs) == 0:
        cols_info = {k: v for k, v in cols_info.items() if col_name == v[0]}
        if len(cols_info) == 0:
            raise ValueError(f"Column {col_name} not found")
        if len(cols_info) > 1:
            raise ValueError(f"Column {col_name} is ambigous")
        return list(cols_info.keys())[0]
    on_df_id = hex(id(on_df))
    cols_info = {
        k: v
        for k, v in cols_info.items()
        if name == v[0] and (v[1].intersection(from_dfs) if from_dfs else True)
    }
    if len(cols_info) == 0:
        if raise_if_not_found:
            raise ValueError(
                f"Column {col_name} not found on DataFrame {on_df}"
            )
        else:
            return False
    if len(cols_info) > 1:
        if raise_if_ambigous:
            raise ValueError(
                f"Column {col_name} is ambigous on DataFrame {on_df}"
            )
        else:
            return True
    return list(cols_info.keys())[0]


def _expr_or_const(c, df):
    if isinstance(c, Column):
        return c._build_expr(df)
    return c


def _name_or_const(c):
    if isinstance(c, Column):
        return c._build_display_name()
    return c


_EXPR_BUILDERS = {
    "__add__": lambda op, df: op[0]._build_expr(df) + _expr_or_const(op[1], df),
    "__sub__": lambda op, df: op[0]._build_expr(df) - _expr_or_const(op[1], df),
    "__mul__": lambda op, df: op[0]._build_expr(df) * _expr_or_const(op[1], df),
    "__truediv__": lambda op, df: op[0]._build_expr(df)
    / _expr_or_const(op[1], df),
    "__mod__": lambda op, df: op[0]._build_expr(df) % _expr_or_const(op[1], df),
    "__pow__": lambda op, df: op[0]._build_expr(df)
    ** _expr_or_const(op[1], df),
    "__neg__": lambda op, df: -op[0]._build_expr(df),
    "__invert__": lambda op, df: ~(op[0]._build_expr(df)),
    "__and__": lambda op, df: op[0]._build_expr(df) & _expr_or_const(op[1], df),
    "__or__": lambda op, df: op[0]._build_expr(df) | _expr_or_const(op[1], df),
    "__gt__": lambda op, df: op[0]._build_expr(df) > _expr_or_const(op[1], df),
    "__ge__": lambda op, df: op[0]._build_expr(df) >= _expr_or_const(op[1], df),
    "__lt__": lambda op, df: op[0]._build_expr(df) < _expr_or_const(op[1], df),
    "__le__": lambda op, df: op[0]._build_expr(df) <= _expr_or_const(op[1], df),
    "__eq__": lambda op, df: op[0]._build_expr(df) == _expr_or_const(op[1], df),
    "__ne__": lambda op, df: op[0]._build_expr(df) != _expr_or_const(op[1], df),
    "isNull": lambda op, df: op[0]._build_expr(df).is_null(),
    "isNotNull": lambda op, df: op[0]._build_expr(df).is_not_null(),
    "between": lambda op, df: op[0]
    ._build_expr(df)
    .is_between(_expr_or_const(op[1], df), _expr_or_const(op[2], df)),
    "startswith": lambda op, df: op[0]
    ._build_expr(df)
    .str.starts_with(_expr_or_const(op[1], df)),
    "endswith": lambda op, df: op[0]
    ._build_expr(df)
    .str.ends_with(_expr_or_const(op[1], df)),
    "substr": lambda op, df: op[0]
    ._build_expr(df)
    .str.slice(op[1] - (1 if op[1] > 0 else 0), op[2]),
    "contains": lambda op, df: op[0]
    ._build_expr(df)
    .str.contains(_expr_or_const(op[1], df)),
    "astype": lambda op, df: op[0]._build_expr(df).cast(types.map_type(op[1])),
    "alias": lambda op, df: op[0]._build_expr(df).alias(op[1]),
    "asc_nulls_first": lambda op, df: op[0]
    ._build_expr(df)
    .sort(descending=False, nulls_last=False),
    "desc_nulls_first": lambda op, df: op[0]
    ._build_expr(df)
    .sort(descending=True, nulls_last=False),
    "asc_nulls_last": lambda op, df: op[0]
    ._build_expr(df)
    .sort(descending=False, nulls_last=True),
    "desc_nulls_last": lambda op, df: op[0]
    ._build_expr(df)
    .sort(descending=True, nulls_last=True),
    "sum": lambda op, df: op[0]._build_expr(df).sum(),
    "count": lambda op, df: op[0]._build_expr(df).count(),
    "avg": lambda op, df: op[0]._build_expr(df).mean(),
    "mean": lambda op, df: op[0]._build_expr(df).mean(),
    "min": lambda op, df: op[0]._build_expr(df).min(),
    "max": lambda op, df: op[0]._build_expr(df).max(),
    "countDistinct": lambda op, df: op[0]._build_expr(df).n_unique(),
}


_EXPR_COL_NAME_BUILDERS = {
    "__add__": lambda op: f"({op[0]._build_display_name()} + {_name_or_const(op[1])})",
    "__sub__": lambda op: f"({op[0]._build_display_name()} - {_name_or_const(op[1])})",
    "__mul__": lambda op: f"({op[0]._build_display_name()} * {_name_or_const(op[1])})",
    "__truediv__": lambda op: f"({op[0]._build_display_name()} / {_name_or_const(op[1])})",
    "__mod__": lambda op: f"({op[0]._build_display_name()} % {_name_or_const(op[1])})",
    "__pow__": lambda op: f"POWER({op[0]._build_display_name()}, {_name_or_const(op[1])})",
    "__neg__": lambda op: f"(- {op[0]._build_display_name()})",
    "__invert__": lambda op: f"(NOT {op[0]._build_display_name()})",
    "__and__": lambda op: f"({op[0]._build_display_name()} AND {_name_or_const(op[1])})",
    "__or__": lambda op: f"({op[0]._build_display_name()} OR {_name_or_const(op[1])})",
    "__gt__": lambda op: f"({op[0]._build_display_name()} > {_name_or_const(op[1])})",
    "__ge__": lambda op: f"({op[0]._build_display_name()} >= {_name_or_const(op[1])})",
    "__lt__": lambda op: f"({op[0]._build_display_name()} < {_name_or_const(op[1])})",
    "__le__": lambda op: f"({op[0]._build_display_name()} <= {_name_or_const(op[1])})",
    "__eq__": lambda op: f"({op[0]._build_display_name()} == {_name_or_const(op[1])})",
    "__ne__": lambda op: f"({op[0]._build_display_name()} != {_name_or_const(op[1])})",
    "isNull": lambda op: f"({op[0]._build_display_name()} IS NULL)",
    "isNotNull": lambda op: f"({op[0]._build_display_name()} IS NOT NULL)",
    "between": lambda op: f"(({op[0]._build_display_name()} >= {_name_or_const(op[1])}) AND ({op[0]._build_display_name()} <= {_name_or_const(op[2])}))",
    "startswith": lambda op: f"startswith({op[0]._build_display_name()}, {_name_or_const(op[1])})",
    "endswith": lambda op: f"endswith({op[0]._build_display_name()}, {_name_or_const(op[1])})",
    "substr": lambda op: f"substring({op[0]._build_display_name()}, {_name_or_const(op[1])}, {_name_or_const(op[2])})",
    "contains": lambda op: f"contains({op[0]._build_display_name()}, {_name_or_const(op[1])})",
    "astype": lambda op: f"CAST({op[0]._build_display_name()} AS {op[1].simpleString().upper()})",
    "alias": lambda op: op[1],
    "asc_nulls_first": lambda op: f"{op[0]._build_display_name()} ASC NULLS FIRST",
    "desc_nulls_first": lambda op: f"{op[0]._build_display_name()} DESC NULLS FIRST",
    "asc_nulls_last": lambda op: f"{op[0]._build_display_name()} ASC NULLS LAST",
    "desc_nulls_last": lambda op: f"{op[0]._build_display_name()} DESC NULLS LAST",
    "sum": lambda op: f"sum({op[0]._build_display_name()})",
    "count": lambda op: f"count({op[0]._build_display_name()})",
    "avg": lambda op: f"avg({op[0]._build_display_name()})",
    "mean": lambda op: f"avg({op[0]._build_display_name()})",
    "min": lambda op: f"min({op[0]._build_display_name()})",
    "max": lambda op: f"max({op[0]._build_display_name()})",
    "countDistinct": lambda op: f"count({op[0]._build_display_name()})",
}


class Column:
    def __init__(
        self,
        name: str,
        on_df=None,
        op=None,
    ):
        self._name = name if on_df is None else _joined_col_name(name, on_df)
        self._on_df = on_df
        self._op = op

    def _build_expr(self, for_df, with_alias=True):
        if self._op is None:
            return pl.col(_identify_column(self._name, for_df))
        e = _EXPR_BUILDERS[self._op[-1]](self._op, for_df)
        if with_alias:
            e = e.alias(self._build_display_name())
        return e

    def _build_display_name(self):
        if self._op is None:
            return _extract_name_and_dfs_from_col_name(self._name)[0]
        n = _EXPR_COL_NAME_BUILDERS[self._op[-1]](self._op)
        return n

    def asc(self):
        return self.asc_nulls_first()

    def desc(self):
        return self.desc_nulls_last()

    def asc_nulls_first(self):
        name = self._build_display_name() + " ASC NULLS FIRST"
        return Column(
            name,
            op=(self, "asc_nulls_first"),
        )

    def desc_nulls_first(self):
        name = self._name + " DESC NULLS FIRST"
        return Column(
            name,
            op=(self, "desc_nulls_first"),
        )

    def asc_nulls_last(self):
        name = self._name + " ASC NULLS LAST"
        return Column(
            name,
            op=(self, "asc_nulls_last"),
        )

    def desc_nulls_last(self):
        name = self._name + " DESC NULLS LAST"
        return Column(
            name,
            op=(self, "desc_nulls_last"),
        )

    def alias(self, name: str):
        return Column(
            name,
            op=(self, name, "alias"),
        )

    def __add__(self, other):
        if isinstance(other, Column):
            name = f"({self._build_display_name()} + {other._build_display_name()})"
        else:
            name = f"({self._name} + {other})"
        return Column(
            name,
            op=(self, other, "__add__"),
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} - {other._name})"
        else:
            name = f"({self._name} - {other})"
        return Column(
            name,
            op=(self, other, "__sub__"),
        )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} * {other._name})"
        else:
            name = f"({self._name} * {other})"
        return Column(
            name,
            op=(self, other, "__mul__"),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} / {other._name})"
        else:
            name = f"({self._name} / {other})"
        return Column(
            name,
            op=(self, other, "__truediv__"),
        )

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __mod__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} % {other._name})"
        else:
            name = f"({self._name} % {other})"
        return Column(
            name,
            op=(self, other, "__mod__"),
        )

    def __rmod__(self, other):
        return self.__mod__(other)

    def __pow__(self, other):
        if isinstance(other, Column):
            name = f"POWER({self._name}, {other._name})"
        else:
            name = f"POWER({self._name}, {other})"
        return Column(
            name,
            op=(self, other, "__pow__"),
        )

    def __rpow__(self, other):
        return self.__pow__(other)

    def __neg__(self):
        name = f"(- {self._name})"
        return Column(
            name,
            op=(self, "__neg__"),
        )

    def __invert__(self):
        name = f"(NOT {self._name})"
        return Column(
            name,
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
        else:
            name = f"startswith({self._name}, {other})"
        return Column(
            name,
            op=(self, other, "startswith"),
        )

    def endswith(self, other):
        if isinstance(other, Column):
            name = f"endswith({self._name}, {other._name})"
        else:
            name = f"endswith({self._name}, {other})"
        return Column(
            name,
            op=(self, other, "endswith"),
        )

    def substr(self, start, length):
        if isinstance(start, Column) or isinstance(length, Column):
            raise ValueError("start and length must be integers")
        if start < 0 or length < 0:
            raise ValueError("start and length must be positive integers")
        name = f"substring({self._name}, {start}, {length})"
        return Column(
            name,
            op=(self, start, length, "substr"),
        )

    def contains(self, other):
        if isinstance(other, Column):
            name = f"contains({self._name}, {other._name})"
        else:
            name = f"contains({self._name}, {other})"
        return Column(
            name,
            op=(self, other, "contains"),
        )

    def astype(self, dtype: Union[types.DataType, str]):
        if isinstance(dtype, str):
            dtype = types._type_name_to_dtype[dtype]
        cname = f"CAST({self._name} AS {dtype.simpleString().upper()})"
        return Column(
            cname,
            op=(self, dtype, "astype"),
        )

    def cast(self, dtype: Union[types.DataType, str]):
        return self.astype(dtype)

    def between(self, lowerBound, upperBound):
        _lowerBound_name = (
            lowerBound._build_display_name()
            if isinstance(lowerBound, Column)
            else str(lowerBound)
        )
        _upperBound_name = (
            upperBound._build_display_name()
            if isinstance(upperBound, Column)
            else str(upperBound)
        )
        return Column(
            f"(({self._name} >= {_lowerBound_name}) AND ({self._name} <= {_upperBound_name}))",
            op=(self, lowerBound, upperBound, "between"),
        )

    def __and__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} AND {other._name})"
        else:
            name = f"({self._name} AND {other})"
        return Column(
            name,
            op=(self, other, "__and__"),
        )

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} OR {other._name})"
        else:
            name = f"({self._name} OR {other})"
        return Column(
            name,
            op=(self, other, "__or__"),
        )

    def __ror__(self, other):
        return self.__or__(other)

    def __gt__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} > {other._name})"
        else:
            name = f"({self._name} > {other})"
        return Column(
            name,
            op=(self, other, "__gt__"),
        )

    def __ge__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} >= {other._name})"
        else:
            name = f"({self._name} >= {other})"
        return Column(
            name,
            op=(self, other, "__ge__"),
        )

    def __lt__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} < {other._name})"
        else:
            name = f"({self._name} < {other})"
        return Column(
            name,
            op=(self, other, "__lt__"),
        )

    def __le__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} <= {other._name})"
        else:
            name = f"({self._name} <= {other})"
        return Column(
            name,
            op=(self, other, "__le__"),
        )

    def __eq__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} == {other._name})"
        else:
            name = f"({self._name} == {other})"
        return Column(name, op=(self, other, "__eq__"))

    def __ne__(self, other):
        if isinstance(other, Column):
            name = f"({self._name} != {other._name})"
        else:
            name = f"({self._name} != {other})"
        return Column(
            name,
            op=(self, other, "__ne__"),
        )

    def isNull(self):
        return Column(
            f"({self._name} IS NULL)",
            op=(self, "isNull"),
        )

    def isNotNull(self):
        return Column(
            f"({self._name} IS NOT NULL)",
            op=(self, "isNotNull"),
        )
