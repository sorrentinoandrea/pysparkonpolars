from typing import Union
import re
import polars as pl
from pyspark.sql import Row
from pysparkpl.sql.functions import col


class DataFrame:
    def __init__(self, df: Union[pl.DataFrame, pl.LazyFrame]):
        self._df: pl.LazyFrame = (
            df if isinstance(df, pl.LazyFrame) else df.lazy()
        )

    _IS_SIMPLE_CAST = re.compile(
        r"^CAST\(([a-zA-Z0-9_-]+) AS [A-Z]+[A-Z0-9_]*\)$"
    )

    def select(self, *cols: Union[str, col]):
        pl_df = self._df.select(
            *[c._expr if isinstance(c, col) else c for c in cols]
        )
        rename_map = {}
        for c in pl_df.columns:
            m = self._IS_SIMPLE_CAST.match(c)
            if m:
                rename_map[c] = m.group(1)
        pl_df = pl_df.rename(rename_map)
        return DataFrame(pl_df)

    def filter(self, condition: col):
        return DataFrame(self._df.filter(condition._expr))

    def collect(self):
        row = Row(*self._df.columns)
        return [row(*r) for r in self._df.collect()]

    def __getattr__(self, name: str):
        return col(name)
