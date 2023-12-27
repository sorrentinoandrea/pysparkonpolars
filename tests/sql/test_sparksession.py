import polars as pl
import pyspark.sql as pssql
import pysparkpl.sql.types as types
import pyspark as ps
from pysparkpl.sql.sparksession import SparkSession
from tests.utils import spark_to_pl


def test_sparksession():
    spark = ps.sql.SparkSession.builder.getOrCreate()
    sprkpl = SparkSession()
    df = sprkpl.createDataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    ps_df = spark.createDataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert df._df.collect().frame_equal(
        pl.DataFrame({"a": [1, 3], "b": [2, 4]})
    )
    assert df._df.collect().frame_equal(spark_to_pl(ps_df))

    pl_df = pl.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    df = sprkpl.createDataFrame(pl_df)
    assert df._df.collect().frame_equal(pl_df)

    pd_df = pl_df.to_pandas()
    df = sprkpl.createDataFrame(pd_df)
    ps_df = spark.createDataFrame(pd_df)
    assert df._df.collect().frame_equal(pl_df)
    assert df._df.collect().frame_equal(spark_to_pl(ps_df))

    df = sprkpl.createDataFrame([[1, 2], [3, 4]], schema=["a", "b"])
    sp_df = spark.createDataFrame([[1, 2], [3, 4]], schema=["a", "b"])
    assert df._df.collect().frame_equal(
        pl.DataFrame({"a": [1, 3], "b": [2, 4]})
    )
    assert df._df.collect().frame_equal(spark_to_pl(sp_df))

    df = sprkpl.createDataFrame(
        [[1, 2], [3, 4]],
        schema=types.StructType(
            [
                types.StructField("a", types.IntegerType()),
                types.StructField("b", types.IntegerType()),
            ]
        ),
    )
    sp_df = spark.createDataFrame(
        [[1, 2], [3, 4]],
        schema=types.StructType(
            [
                types.StructField("a", types.IntegerType()),
                types.StructField("b", types.IntegerType()),
            ]
        ),
    )
    assert df._df.collect().frame_equal(
        pl.DataFrame(
            {"a": [1, 3], "b": [2, 4]},
            schema=pl.Struct(
                [pl.Field("a", pl.Int32), pl.Field("b", pl.Int32)]
            ),
        )
    )
    assert df._df.collect().frame_equal(spark_to_pl(sp_df))

    df = sprkpl.createDataFrame(
        [[1, 2], [3, 4]],
    )
    sp_df = spark.createDataFrame(
        [[1, 2], [3, 4]],
    )
    assert df._df.collect().frame_equal(
        pl.DataFrame({"_1": [1, 3], "_2": [2, 4]})
    )
    assert df._df.collect().frame_equal(spark_to_pl(sp_df))

    row = pssql.Row("a", "b")
    df = sprkpl.createDataFrame([row(1, 2), row(3, 4)])
    sp_df = spark.createDataFrame([row(1, 2), row(3, 4)])
    assert df._df.collect().frame_equal(
        pl.DataFrame({"a": [1, 3], "b": [2, 4]})
    )
    assert df._df.collect().frame_equal(spark_to_pl(sp_df))


if __name__ == "__main__":
    test_sparksession()
