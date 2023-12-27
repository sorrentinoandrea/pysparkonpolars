from typing import Union, Optional, Iterable, Any, TYPE_CHECKING
import os

print(os.environ["PYTHONPATH"])
import polars as pl
import pandas as pd

from pysparkpl.sql.types import AtomicType, StructType
from pysparkpl.sql.dataframe import DataFrame
from pysparkpl.sql.types import map_type

import pyspark.sql as pssql

if TYPE_CHECKING:
    from pyspark.sql.pandas.types import (
        DataFrameLike as PandasDataFrameLike,
        ArrayLike,
    )


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
        if isinstance(data, pl.DataFrame):
            return DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return DataFrame(pl.from_pandas(data))
        elif isinstance(data, Iterable):
            schema = (
                schema
                if not isinstance(schema, StructType)
                else map_type(schema)
            )
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
            return DataFrame(pl.LazyFrame(data, schema=schema, orient="row"))
        else:
            raise NotImplementedError(
                "Only pandas and polars and records are supported"
            )

    class builder:
        @staticmethod
        def getOrCreate():
            return SparkSession()
