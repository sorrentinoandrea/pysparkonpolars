from typing import Union, Dict, Callable, List, Tuple, Set
import re
from collections import Iterable
import polars as pl
from pyspark.sql import Row
from pysparkpl.sql.functions import col
from pysparkpl.sql.column import (
    _identify_column,
    _extract_name_and_dfs_from_col_name,
    _joined_col_name,
)
import pysparkpl.sql.types as types
import pyspark.sql.types as pssql_types

"""
A pyspark-like DataFrame API on top of polars.
"""


class GroupedData(object):
    """
    A set of methods for aggregations on a :class:`DataFrame`, created by
    :func:`DataFrame.
    """

    def __init__(self, group_by: pl.dataframe.groupby.GroupBy, df: "DataFrame"):
        self._group_by = group_by
        self._df = df

    def __aggop(
        self,
        op_name: str,
        target_op_name: str,
        col_type_validator: Callable[[str], bool],
    ):
        def _aggop(*cols: str):
            _cols = (
                [
                    getattr(
                        pl.col(_identify_column(c, self._df)), op_name
                    )().alias(f"{target_op_name}({c})")
                    for c in cols
                ]
                if cols
                else [
                    getattr(pl.col(c), op_name)().alias(
                        f"{target_op_name}({_extract_name_and_dfs_from_col_name(c)[0]})"
                    )
                    for c in self._df._df.schema
                    if col_type_validator(self._df._df.schema[c])
                ]
            )

            return DataFrame(self._group_by.agg(*_cols))

        return _aggop

    def sum(self, *cols: str):
        return self.__aggop("sum", "sum", types.pl_is_numeric_type)(*cols)

    def count(self):
        return DataFrame(self._group_by.count())

    def avg(self, *cols: str):
        return self.__aggop("mean", "avg", types.pl_is_numeric_type)(*cols)

    def mean(self, *cols: str):
        return self.avg(*cols)

    def min(self, *cols: str):
        return self.__aggop("min", "min", types.pl_is_numeric_type)(*cols)

    def max(self, *cols: str):
        return self.__aggop("max", "max", types.pl_is_numeric_type)(*cols)

    def agg(self, *map_or_cols: Union[Dict[str, str], col]):
        if len(map_or_cols) == 0:
            raise ValueError("agg() missing 1 required positional argument")
        if len(map_or_cols) == 1 and isinstance(map_or_cols[0], dict):
            map_or_cols = [
                col(k).getattr(v).alias(f"{v}({k})")
                for k, v in map_or_cols[0].items()
            ]
        else:
            if any(not isinstance(c, col) for c in map_or_cols):
                raise ValueError(
                    "all arguments of aggregate function should be Column"
                )

        return DataFrame(
            self._group_by.agg(*[c._build_expr(self._df) for c in map_or_cols])
        )


class DataFrame:
    def __init__(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        column_names_map: Dict[str, str] = None,
    ):
        self.column_names_map = column_names_map or {}
        self._df: pl.LazyFrame = (
            df if isinstance(df, pl.LazyFrame) else df.lazy()
        )
        self._df = self._df.rename(
            {c: _joined_col_name(c, self) for c in df.columns}
        )

    _IS_SIMPLE_CAST = re.compile(
        r"^CAST\(([a-zA-Z0-9_-]+) AS [A-Z]+[A-Z0-9_]*\)$"
    )

    def select(self, *cols: Union[str, col]):
        _cols = [
            col(_identify_column(c, self)) if isinstance(c, str) else c
            for c in cols
        ]

        pl_df = self._df.select(*[c._build_expr(self) for c in _cols])
        rename_map = {}
        for c in pl_df.columns:
            m = self._IS_SIMPLE_CAST.match(c)
            if m:
                rename_map[c] = m.group(1)
        pl_df = pl_df.rename(rename_map)
        return DataFrame(pl_df)

    def filter(self, condition: col):
        return DataFrame(self._df.filter(condition._build_expr(self)))

    def collect(self):
        row = Row(*self._df.columns)
        df = self.toPandas()
        return [row(*r) for r in df.to_records(index=False)]

    def toPandas(self):
        df = self._df.collect().to_pandas()
        df.columns = [
            _extract_name_and_dfs_from_col_name(c)[0] for c in df.columns
        ]
        return df

    def groupBy(self, *cols: Union[str, col]):
        return GroupedData(
            self._df.groupby(
                *[c._build_expr(self) for c in cols], maintain_order=True
            ),
            self,
        )

    def count(self):
        return self._df.collect().shape[0]

    def withColumn(self, colName: str, col: col):
        return DataFrame(
            self._df.with_columns(col._build_expr(self).alias(colName))
        )

    def withColumns(self, *colsMap: Dict[str, col]):
        if len(colsMap) != 1:
            raise ValueError("withColumns() takes exactly 1 argument")
        colsMap = colsMap[0]
        return DataFrame(
            self._df.with_columns(
                *[c._build_expr(self).alias(n) for n, c in colsMap.items()]
            )
        )

    def withColumnRenamed(self, existingName: str, newName: str):
        _existingName = _identify_column(existingName, self)
        return DataFrame(
            self._df.rename({_existingName: newName}),
        )

    def sort(
        self, *cols: Union[str, col], ascending: Union[bool, List[bool]] = None
    ):
        if ascending is None:
            descending = False
        elif isinstance(ascending, bool):
            descending = not ascending
        else:
            if not isinstance(ascending, Iterable):
                raise ValueError(
                    "ascending can only be a boolean or an Iterable of booleans"
                )
            if not hasattr(ascending, "__len__"):
                ascending = list(ascending)
            if len(ascending) != len(cols):
                raise ValueError(
                    "ascending should have the same length as cols"
                )
            descending = (
                [not a for a in ascending]
                if len(ascending) != 1
                else not ascending[0]
            )

        return DataFrame(
            self._df.sort(
                *[
                    c._build_expr(self)
                    if isinstance(c, col)
                    else _identify_column(c, self)
                    for c in cols
                ],
                descending=descending,
            )
        )

    def join(
        self,
        other: "DataFrame",
        on: Union[str, col, List[Union[str, col]]] = None,
        how="inner",
    ):
        if isinstance(on, col):
            return self._join_on_expr(other, on, how)

        _EQUIVALENT_HOW = {
            "full": "outer",
            "full_outer": "outer",
            "fullouter": "outer",
            "right_outer": "right",
            "rightouter": "right",
            "left_outer": "left",
            "leftouter": "left",
            "left_semi": "semi",
            "leftsemi": "semi",
            "left_anti": "anti",
            "leftanti": "anti",
        }

        how = _EQUIVALENT_HOW.get(how, how)

        if on is None:
            on = []
        if isinstance(on, str):
            on = [on]
        if isinstance(on, Iterable):
            on = list(on) if not isinstance(on, (list, tuple)) else on
            if not all([isinstance(c, (str)) for c in on]):
                raise ValueError(
                    "on should be a string, a list or an Iterable of strings or an expression"
                )

        left_on = [_identify_column(c, on_df=self) for c in on]
        right_on = [_identify_column(c, on_df=other) for c in on]

        if how == "right" and on:
            joined = other._df.join(
                self._df,
                left_on=right_on,
                right_on=left_on,
                how="left",
            )
        else:
            if on:
                joined = self._df.join(
                    other._df, left_on=left_on, right_on=right_on, how=how
                )
            elif how not in ["semi", "anti"]:
                joined = self._df.join(other._df, how="cross")
            elif how == "semi":
                joined = self._df
            else:  # how == "anti"
                joined = pl.DataFrame([], schema=self._df.schema)
        if how == "inner" and len(left_on):
            joined = joined.rename(
                {c: _joined_col_name(c, other) for c in left_on}
            )
        return DataFrame(
            joined,
            column_names_map={},
        )

    def _join_on_expr(self, other: "DataFrame", on: col, how="inner"):
        try:
            (
                left_on,
                right_on,
                left_condition,
                right_condition,
            ) = _breakdown_on_expr(on, self, other)
        except UnsupportedJoinConditionError:
            return self._join_on_expr_with_cross_join(other, on, how)
        dfl = self if left_condition is None else self.filter(left_condition)
        dfr = (
            other if right_condition is None else other.filter(right_condition)
        )
        if len(left_on) == 0:
            if how not in ["semi", "anti"]:
                joined = dfl._df.join(dfr._df, how="cross")
            elif how == "semi":
                joined = dfl._df
            else:
                joined = pl.DataFrame([], schema=dfl.schema)
        else:
            dfl_df = dfl._df.with_columns(
                [
                    c._build_expr(dfl).alias(f"join_col_{i}")
                    for i, c in enumerate(left_on)
                ]
            )
            dfr_df = dfr._df.with_columns(
                [
                    c._build_expr(dfr).alias(f"join_col_{i}")
                    for i, c in enumerate(right_on)
                ]
            )
            if how == "right":
                joined = dfr_df.join(
                    dfl_df,
                    [f"join_col_{i}" for i in range(len(right_on))],
                    how="left",
                ).drop([f"join_col_{i}" for i in range(len(right_on))])
            else:
                joined = dfl_df.join(
                    dfr_df,
                    [f"join_col_{i}" for i in range(len(left_on))],
                    how=how,
                ).drop([f"join_col_{i}" for i in range(len(left_on))])
        return DataFrame(joined)

    def _join_on_expr_with_cross_join(self, other, on: col, how="inner"):
        return self.join(other, how="cross").filter(on)

    @property
    def columns(self):
        return [
            _extract_name_and_dfs_from_col_name(c)[0] for c in self._df.columns
        ]

    @property
    def schema(self):
        fields = []
        if self.column_names_map:
            for c in self._df.schema:
                fields.append(
                    pssql_types.StructField(
                        self.column_names_map.get(c, c),
                        types.reverse_type_mapping[self._df.schema[c]],
                        True,
                    ),
                )
        else:
            for c in self._df.schema:
                fields.append(
                    pssql_types.StructField(
                        c,
                        types.reverse_type_mapping[self._df.schema[c]],
                        True,
                    )
                )
        return pssql_types.StructType(fields)

    def __getattr__(self, name: str):
        return col(name, on_df=self)

    def __getitem__(self, item: Union[str, col]):
        if isinstance(item, str):
            return col(item, on_df=self)
        if isinstance(item, col):
            return col(item._name, on_df=self)
        raise ValueError("item should be a string or a Column")


def _extract_columns(column: col) -> List[col]:
    rv = []
    if not isinstance(column, col):
        return []
    if column._op is None:
        return [column]
    for c in column._op[:-1]:
        rv.extend(_extract_columns(c))

    return rv


def _is_expr_on_df(e: col, df: DataFrame):
    if e._op is None:
        return bool(
            _identify_column(
                e._name, df, raise_if_not_found=False, raise_if_ambigous=False
            )
        )
    matches = []
    for c in e._op[:-1]:
        matches.append(isinstance(c, col) and _is_expr_on_df(c, df))
    return any(matches)


class UnsupportedJoinConditionError(Exception):
    pass


def _breakdown_on_expr(e: col, dfl, dfr):
    left_on = []
    right_on = []
    left_conditions = []
    right_conditions = []

    if e._op is None:
        raise UnsupportedJoinConditionError(
            "Join conditions involving both dataframes must be equality conditions, or & of single dataframe conditions or equality conditions"
        )
    conditions = []
    if e._op[-1] == "__and__":
        conditions += [e._op[0], e._op[1]]
        and_found = True
        while and_found:
            and_found = False
            for i, c in enumerate(conditions):
                if c._op[-1] == "__and__":
                    conditions += [c._op[0], c._op[1]]
                    conditions.pop(i)
                    and_found = True
                    break
    else:
        conditions += [e]

    for c in conditions:
        if _is_expr_on_df(e, dfl) and not _is_expr_on_df(e, dfr):
            left_conditions.append(c)
        elif _is_expr_on_df(e, dfr) and not _is_expr_on_df(e, dfl):
            right_conditions.append(c)
        else:
            if c._op[-1] != "__eq__":
                raise UnsupportedJoinConditionError(
                    "Conditions involving both dataframes must be equality conditions, or & of single dataframe conditions or equality conditions"
                )
            if (
                _is_expr_on_df(c._op[0], dfl)
                and (not _is_expr_on_df(c._op[0], dfr))
                and (_is_expr_on_df(c._op[1], dfr))
                and (not _is_expr_on_df(c._op[1], dfl))
            ):
                left_on.append(c._op[0])
                right_on.append(c._op[1])
            elif (
                _is_expr_on_df(c._op[1], dfl)
                and (not _is_expr_on_df(c._op[1], dfr))
                and (_is_expr_on_df(c._op[0], dfr))
                and (not _is_expr_on_df(c._op[0], dfl))
            ):
                left_on.append(c._op[1])
                right_on.append(c._op[0])
            else:
                raise UnsupportedJoinConditionError(
                    "Conditions involving both dataframes must be equality conditions, or & of single dataframe conditions or equality conditions"
                )
    left_condition = None if len(left_conditions) == 0 else left_conditions[0]
    for c in left_conditions[1:]:
        left_condition = left_condition & c
    right_condition = (
        None if len(right_conditions) == 0 else right_conditions[0]
    )
    for c in right_conditions[1:]:
        right_condition = right_condition & c
    return left_on, right_on, left_condition, right_condition
