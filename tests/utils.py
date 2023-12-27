import polars as pl


def spark_to_pl(df):
    return pl.DataFrame(df.toPandas())
