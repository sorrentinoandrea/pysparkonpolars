from datetime import date
import polars as pl
import pyspark as ps
import pysparkpl as pspl
import pysparkpl.sql.dataframe as dataframe
import pyspark.sql.types as pstypes
from tests.utils import spark_to_pl


def test_dataframe_select():
    df = dataframe.DataFrame(pl.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}))
    assert (
        df.select("a")._df.collect().frame_equal(pl.DataFrame({"a": [1, 2, 3]}))
    )
    assert (
        df.select(df.a)
        ._df.collect()
        .frame_equal(pl.DataFrame({"a": [1, 2, 3]}))
    )
    assert (
        df.select(df.a, df.b)
        ._df.collect()
        .frame_equal(pl.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}))
    )
    assert (
        df.select(df.a + 1)
        ._df.collect()
        .frame_equal(pl.DataFrame({"(a + 1)": [2, 3, 4]}))
    )
    assert (
        df.select(df.a + df.b)
        ._df.collect()
        .frame_equal(pl.DataFrame({"(a + b)": [3, 5, 7]}))
    )
    assert (
        df.select(df.a + df.b + 1)
        ._df.collect()
        .frame_equal(pl.DataFrame({"((a + b) + 1)": [4, 6, 8]}))
    )
    assert (
        df.select(df.a + df.b + df.a)
        ._df.collect()
        .frame_equal(pl.DataFrame({"((a + b) + a)": [4, 7, 10]}))
    )
    assert (
        df.select(df.a.alias("a1"), df.b.alias("b1"))
        ._df.collect()
        .frame_equal(pl.DataFrame({"a1": [1, 2, 3], "b1": [2, 3, 4]}))
    )


def create_test_df_1(spark):
    return spark.createDataFrame(
        [
            {"a": 1, "b": 2, "d": date(2023, 12, 27)},
            {"a": 2, "b": 3, "d": date(2023, 12, 28)},
            {"a": 3, "b": 4, "d": date(2023, 12, 29)},
        ],
        schema=["a", "b", "d"],
    )


def _test_select(spark):
    df = create_test_df_1(spark)
    rv = []
    rv.append(df.select(df.a + df.b))
    rv.append(df.select(df.a + df.b + 1))
    rv.append(df.select(df.a + df.b + df.a))
    rv.append(df.select(df.a - df.b))
    rv.append(df.select(df.a - df.b - 1))
    rv.append(df.select(df.a - df.b - df.a))
    rv.append(df.select(df.a * df.b))
    rv.append(df.select(df.a * df.b * 1))
    rv.append(df.select(df.a * df.b * df.a))
    rv.append(df.select(df.a / df.b))
    rv.append(df.select(df.a / df.b / 1))
    rv.append(df.select(df.a / df.b / df.a))

    rv.append(df.select(df.a + 1, df.b + 1))
    rv.append(df.select(df.a + 1, df.b + 1, df.a + 2, df.b + 2))

    rv.append(df.select(df.a % df.b))
    rv.append(df.select(df.a % df.b % 1))
    rv.append(df.select(df.a % df.b % df.a))

    rv.append(df.select(df.a.alias("a1"), df.b.alias("b1")))
    rv.append(df.select(df.a.alias("a1"), df.b.alias("b1"), df.a.alias("a2")))
    rv.append(
        df.select((df.a + 1).alias("Result"), df.b + 1, df.a + 2, df.b + 2)
    )

    rv.append(df.select(df.a.astype("boolean") & df.b.astype("boolean")))
    rv.append(
        df.select(
            df.a.astype("boolean")
            & df.b.astype("boolean")
            & df.a.astype("boolean")
        )
    )
    rv.append(df.select(df.a.astype("boolean") | df.b.astype("boolean")))
    rv.append(
        df.select(
            df.a.astype("boolean")
            | df.b.astype("boolean")
            | df.a.astype("boolean")
        )
    )

    rv.append(df.select(df.a.astype("int")))
    rv.append(df.select(df.a.astype("int") + 1))
    rv.append(df.select((df.a + 1).astype("int")))
    rv.append(df.select((df.a + 1).astype("string")))
    rv.append(df.select(df.a.astype("float")))
    rv.append(df.select(df.a.astype("double")))
    rv.append(df.select(df.a.astype("boolean")))
    rv.append(df.select(df.d.astype("date")))

    rv.append(df.select(df.a.astype(pstypes.IntegerType())))
    rv.append(df.select(df.a.astype(pstypes.IntegerType()) + 1))
    rv.append(df.select((df.a + 1).astype(pstypes.IntegerType())))
    rv.append(df.select((df.a + 1).astype(pstypes.StringType())))
    rv.append(df.select(df.a.astype(pstypes.FloatType())))
    rv.append(df.select(df.a.astype(pstypes.DoubleType())))
    rv.append(df.select(df.a.astype(pstypes.BooleanType())))
    rv.append(df.select(df.d.astype(pstypes.DateType())))

    return rv


def _test_filter(spark):
    df = create_test_df_1(spark)
    rv = []
    rv.append(df.filter(df.a > 1))
    rv.append(df.filter(df.a > df.b))
    rv.append(df.filter(df.a > df.b + 1))
    rv.append(df.filter(df.a > df.b + df.a))
    rv.append(df.filter(df.a > df.b - 1))
    rv.append(df.filter(df.a > df.b - df.a))

    rv.append(df.filter(df.a < 1))
    rv.append(df.filter(df.a < df.b))
    rv.append(df.filter(df.a < df.b + 1))
    rv.append(df.filter(df.a < df.b + df.a))
    rv.append(df.filter(df.a < df.b - 1))
    rv.append(df.filter(df.a < df.b - df.a))

    rv.append(df.filter(df.a >= 1))
    rv.append(df.filter(df.a >= df.b))
    rv.append(df.filter(df.a >= df.b + 1))
    rv.append(df.filter(df.a >= df.b + df.a))
    rv.append(df.filter(df.a >= df.b - 1))
    rv.append(df.filter(df.a >= df.b - df.a))

    rv.append(df.filter(df.a <= 1))
    rv.append(df.filter(df.a <= df.b))
    rv.append(df.filter(df.a <= df.b + 1))
    rv.append(df.filter(df.a <= df.b + df.a))
    rv.append(df.filter(df.a <= df.b - 1))
    rv.append(df.filter(df.a <= df.b - df.a))

    rv.append(df.filter(df.a == 1))
    rv.append(df.filter(df.a == df.b))
    rv.append(df.filter(df.a == df.b + 1))
    rv.append(df.filter(df.a == df.b + df.a))
    rv.append(df.filter(df.a == df.b - 1))
    rv.append(df.filter(df.a == df.b - df.a))

    rv.append(df.filter(df.a != 1))
    rv.append(df.filter(df.a != df.b))
    rv.append(df.filter(df.a != df.b + 1))
    rv.append(df.filter(df.a != df.b + df.a))
    rv.append(df.filter(df.a != df.b - 1))
    rv.append(df.filter(df.a != df.b - df.a))

    rv.append(df.filter((df.a > 1) & (df.b > 1)))
    rv.append(df.filter((df.a > 1) & (df.b > df.a)))
    rv.append(df.filter((df.a > 1) & (df.b > df.a + 1)))
    rv.append(df.filter((df.a > 1) & (df.b > df.a + df.b)))
    rv.append(df.filter((df.a > 1) & (df.b > df.a - 1)))
    rv.append(df.filter((df.a > 1) & (df.b > df.a - df.b)))

    rv.append(df.filter((df.a > 1) | (df.b > 1)))
    rv.append(df.filter((df.a > 1) | (df.b > df.a)))
    rv.append(df.filter((df.a > 1) | (df.b > df.a + 1)))
    rv.append(df.filter((df.a > 1) | (df.b > df.a + df.b)))
    rv.append(df.filter((df.a > 1) | (df.b > df.a - 1)))
    rv.append(df.filter((df.a > 1) | (df.b > df.a - df.b)))

    return rv


def test_filter():
    sparkpl = pspl.sql.SparkSession.builder.getOrCreate()
    spark = ps.sql.SparkSession.builder.getOrCreate()
    pl_results = _test_filter(sparkpl)
    ps_results = _test_filter(spark)
    for i in range(len(pl_results)):
        assert (
            pl_results[i]._df.collect().frame_equal(spark_to_pl(ps_results[i]))
        )


def test_select():
    sparkpl = pspl.sql.SparkSession.builder.getOrCreate()
    spark = ps.sql.SparkSession.builder.getOrCreate()
    pl_results = _test_select(sparkpl)
    ps_results = _test_select(spark)
    for i in range(len(pl_results)):
        assert (
            pl_results[i]._df.collect().frame_equal(spark_to_pl(ps_results[i]))
        )


if __name__ == "__main__":
    test_dataframe_select()
    test_select()
    test_filter()
