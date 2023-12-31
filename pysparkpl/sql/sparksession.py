from typing import Union, Optional, Iterable, Any, TYPE_CHECKING
from enum import Enum
import os

import polars as pl
import pandas as pd

import pyspark as ps
import pysparkpl as pspl

from pysparkpl.sql.types import AtomicType, StructType, StructField
from pysparkpl.sql.dataframe import DataFrame
from pysparkpl.sql.types import map_type

import pyspark.sql as pssql

if TYPE_CHECKING:
    from pyspark.sql.pandas.types import (
        DataFrameLike as PandasDataFrameLike,
        ArrayLike,
    )


def make_names_unique(schema: Union[Iterable[str], StructType]):
    seen = {}
    names_map = {}
    at_least_one_duplicate = False
    if isinstance(schema, StructType):
        fields = []
        for field in schema:
            if field.name in seen:
                at_least_one_duplicate = True
                new_name = f"{field.name}_duplicated_{seen[field.name]}"
                seen[field.name] += 1
                names_map[new_name] = field.name
            else:
                seen[field.name] = 1
                names_map[field.name] = field.name
                new_name = field.name
            fields.append(StructField(new_name, field.dataType, field.nullable))
        return StructType(fields), names_map if at_least_one_duplicate else {}

    names = list(schema)
    for i, name in enumerate(names):
        if name in seen:
            at_least_one_duplicate = True
            names[i] = f"{name}_duplicated_{seen[name]}"
            seen[name] += 1
            names_map[names[i]] = name
        else:
            seen[name] = 1
            names_map[name] = name
    return names, names_map if at_least_one_duplicate else {}


class SparkSession:
    def __init__(self):
        pass

    def createDataFrame(
        self,
        data: Union[Iterable[Any], "PandasDataFrameLike", "ArrayLike"],
        schema: Union[AtomicType, StructType, str, None] = None,
        samplingRatio: Optional[float] = None,
        verifySchema: bool = True,
    ):
        # Pyspark allows duplicated column names, only when a schema is provided
        # It is not possible to have duplicated column names when data is
        # pandas, dicts or records
        names_map = {}
        if schema is not None:
            schema, names_map = make_names_unique(schema)
        if isinstance(schema, StructType):
            schema = map_type(schema)
        if isinstance(data, pl.DataFrame):
            return DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            rv = DataFrame(pl.from_pandas(data))
            return rv
        elif isinstance(data, Iterable):
            if schema is None:
                if not isinstance(data, (list, tuple)):
                    data = list(data)
                if len(data) == 0:
                    raise ValueError("can not infer schema from empty dataset")
                if not isinstance(data[0], dict):
                    schema = [f"_{i+1}" for i in range(len(data[0]))]
                if isinstance(data[0], pssql.Row):
                    return DataFrame(
                        pl.DataFrame._from_dicts([d.asDict() for d in data])
                    )
            else:
                return DataFrame(
                    pl.DataFrame(data, schema=schema, orient="row"),
                    column_names_map=names_map,
                )
            return DataFrame(
                pl.LazyFrame(data, schema=schema, orient="row"),
                column_names_map=names_map,
            )
        else:
            raise NotImplementedError(
                "Only pandas and polars and records are supported"
            )

    class builder:
        @staticmethod
        def getOrCreate():
            return SparkSession()


class SessionHelper:
    class Engine(Enum):
        SPARK = 1
        POLARS = 2

    @classmethod
    def sql(cls, df_or_engine: Union[ps.sql.DataFrame, DataFrame, Engine]):
        if isinstance(df_or_engine, ps.sql.DataFrame) or (
            isinstance(df_or_engine, cls.Engine)
            and df_or_engine == cls.Engine.SPARK
        ):
            return ps.sql
        else:
            return pspl.sql

    @classmethod
    def functions(
        cls, df_or_engine: Union[ps.sql.DataFrame, DataFrame, Engine]
    ):
        if isinstance(df_or_engine, ps.sql.DataFrame) or (
            isinstance(df_or_engine, cls.Engine)
            and df_or_engine == cls.Engine.SPARK
        ):
            return ps.sql.functions
        else:
            return pspl.sql.functions

    @staticmethod
    def session(engine: Engine):
        if engine == SessionHelper.Engine.SPARK:
            return ps.sql.SparkSession
        elif engine == SessionHelper.Engine.POLARS:
            return pspl.sql.SparkSession
        else:
            raise ValueError(f"Unknown engine {engine}")
