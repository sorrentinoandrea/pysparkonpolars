from datetime import date
import pandas as pd
import polars as pl
import pyspark as ps
import pysparkpl as pspl
import pysparkpl.sql.dataframe as dataframe
import pyspark.sql.types as pstypes
from tests.utils import spark_to_pl
from pysparkpl.sql import SessionHelper


def compare_pl_spark(pl_df, ps_df):
    if set(pl_df.columns) != set(ps_df.columns):
        return False
    columns = pl_df.columns
    ps_df = ps_df.select(*pl_df.columns)
    pl_df = pl_df.sort(*columns)
    ps_df = ps_df.sort(*columns)
    return pl_df._df.collect().frame_equal(spark_to_pl(ps_df))


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
            {"a": 1, "b": 2, "c": "string1", "d": date(2023, 12, 27)},
            {"a": 2, "b": 3, "c": "string2", "d": date(2023, 12, 28)},
            {"a": 3, "b": 4, "c": "string3", "d": date(2023, 12, 29)},
        ],
        schema=["a", "b", "c", "d"],
    )


def create_test_df_2(spark):
    a = [int(x / 10) for x in range(100)]
    b = [int(x / 5) for x in range(100)]
    c = [x for x in range(100)]
    return spark.createDataFrame(pd.DataFrame({"a": a, "b": b, "c": c}))


def test_collect():
    spark = SessionHelper.session(
        SessionHelper.Engine.SPARK
    ).builder.getOrCreate()
    sparkpl = SessionHelper.session(
        SessionHelper.Engine.POLARS
    ).builder.getOrCreate()
    sp_df = create_test_df_1(spark).select(["a", "b", "c"])
    pl_df = create_test_df_1(sparkpl).select(["a", "b", "c"])
    assert compare_pl_spark(pl_df, sp_df)
    assert sp_df.collect() == pl_df.collect()


def _test_select(spark):
    df = create_test_df_1(spark)
    rv = []

    rv.append(df.select(df.a))
    rv.append(df.select(-df.a))
    rv.append(df.select(-df.a + 1))
    rv.append(df.select(-df.a + -df.b))

    rv.append(df.select(~df.a.astype("boolean")))

    # rv.append(df.select(abs(df.a))) # unsupported in pyspark

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

    rv.append(df.select(df.a**2))

    rv.append(df.select(df.a.alias("a1"), df.b.alias("b1")))
    rv.append(df.select(df.a.alias("a1"), df.b.alias("b1"), df.a.alias("a2")))
    rv.append(
        df.select((df.a + 1).alias("Result"), df.b + 1, df.a + 2, df.b + 2)
    )

    rv.append(df.select(df.a.astype("int")))
    rv.append(df.select(df.a.astype("int") + 1))
    rv.append(df.select((df.a + 1).astype("int")))

    rv.append(df.select(df.a.astype("float")))
    rv.append(df.select(df.a.astype("double")))
    rv.append(df.select(df.a.astype("string")))
    rv.append(df.select(df.a.astype("boolean")))
    rv.append(df.select(df.d.astype("date")))

    rv.append(df.select(df.a.cast("float")))
    rv.append(df.select(df.a.cast("double")))
    rv.append(df.select(df.a.cast("string")))
    rv.append(df.select(df.a.cast("boolean")))
    rv.append(df.select(df.d.cast("date")))

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

    rv.append(df.select(df.c.startswith("string")))
    rv.append(df.select(df.c.endswith("g1")))
    rv.append(df.select(df.c.contains("ring")))

    rv.append(df.select(df.c.substr(0, 2)))
    rv.append(df.select(df.c.substr(1, 2)))

    rv.append(df.select(df.a.isNull()))
    rv.append(df.select(df.a.isNotNull()))

    rv.append(df.select(df.a.between(1, 2)))
    rv.append(df.select(df.a.between(1, df.b)))

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

    rv.append(df.filter(~(df.a > 3)))
    rv.append(df.filter(~(df.a > df.b)))
    rv.append(df.filter(~(df.a > df.b + 1)))
    rv.append(df.filter(~(df.a > df.b + df.a)))
    rv.append(df.filter(~(df.a > df.b - 1)))
    rv.append(df.filter(~(df.a > df.b - df.a)))

    rv.append(df.filter(~(df.a >= 3)))
    rv.append(df.filter(~(df.a >= df.b)))
    rv.append(df.filter(~(df.a >= df.b + 1)))
    rv.append(df.filter(~(df.a >= df.b + df.a)))
    rv.append(df.filter(~(df.a >= df.b - 1)))
    rv.append(df.filter(~(df.a >= df.b - df.a)))

    rv.append(df.filter(df.c.startswith("string")))
    rv.append(df.filter(df.c.endswith("g1")))
    rv.append(df.filter(df.c.contains("ring")))
    rv.append(df.filter(df.c.startswith("string") & (df.a > 1)))

    rv.append(df.filter(df.c.substr(0, 2) == "st"))
    rv.append(df.filter(df.c.substr(1, 2) == "tr"))

    rv.append(df.filter(df.a.isNull()))
    rv.append(df.filter(df.a.isNotNull()))

    rv.append(df.filter(df.a.between(1, 2)))
    rv.append(df.filter(df.a.between(1, df.b)))

    return rv


def _test_groupBy(spark):
    df = create_test_df_1(spark)
    F = SessionHelper.functions(df)
    rv = []
    rv.append(df.groupBy(df.a).sum())
    rv.append(df.groupBy(df.a).sum("b"))
    rv.append(df.groupBy(df.a).sum("b", "a"))

    rv.append(df.groupBy(df.a).count())

    rv.append(df.groupBy(df.a).avg())
    rv.append(df.groupBy(df.a).avg("b"))
    rv.append(df.groupBy(df.a).avg("b", "a"))

    rv.append(df.groupBy(df.a).mean())
    rv.append(df.groupBy(df.a).mean("b"))
    rv.append(df.groupBy(df.a).mean("b", "a"))

    rv.append(df.groupBy(df.a).min())
    rv.append(df.groupBy(df.a).min("b"))
    rv.append(df.groupBy(df.a).min("b", "a"))

    rv.append(df.groupBy(df.a).max())
    rv.append(df.groupBy(df.a).max("b"))
    rv.append(df.groupBy(df.a).max("b", "a"))

    rv.append(df.groupBy(df.a).agg(F.sum(df.b)))
    rv.append(df.groupBy(df.a).agg(F.sum(df.b), F.sum(df.a)))
    rv.append(df.groupBy(df.a).agg(F.sum(df.b) ** 2))
    rv.append(df.groupBy(df.a).agg(F.sum(df.b**2)))
    rv.append(df.groupBy(df.a).agg(F.sum(df.b + df.a)))

    rv.append(df.groupBy(df.a).agg(F.count(df.b)))
    rv.append(df.groupBy(df.a).agg(F.count(df.b), F.count(df.a)))
    rv.append(df.groupBy(df.a).agg(F.count(df.b) ** 2))

    rv.append(df.groupBy(df.a).agg(F.avg(df.b)))
    rv.append(df.groupBy(df.a).agg(F.avg(df.b), F.avg(df.a)))
    rv.append(df.groupBy(df.a).agg(F.avg(df.b) ** 2))
    rv.append(df.groupBy(df.a).agg(F.avg(df.b**2)))
    rv.append(df.groupBy(df.a).agg(F.avg(df.b + df.a)))

    rv.append(df.groupBy(df.a).agg(F.mean(df.b)))
    rv.append(df.groupBy(df.a).agg(F.mean(df.b), F.mean(df.a)))
    rv.append(df.groupBy(df.a).agg(F.mean(df.b) ** 2))
    rv.append(df.groupBy(df.a).agg(F.mean(df.b**2)))
    rv.append(df.groupBy(df.a).agg(F.mean(df.b + df.a)))

    rv.append(df.groupBy(df.a).agg(F.min(df.b)))
    rv.append(df.groupBy(df.a).agg(F.min(df.b), F.min(df.a)))
    rv.append(df.groupBy(df.a).agg(F.min(df.b) ** 2))
    rv.append(df.groupBy(df.a).agg(F.min(df.b**2)))
    rv.append(df.groupBy(df.a).agg(F.min(df.b + df.a)))

    rv.append(df.groupBy(df.a).agg(F.max(df.b)))
    rv.append(df.groupBy(df.a).agg(F.max(df.b), F.max(df.a)))
    rv.append(df.groupBy(df.a).agg(F.max(df.b) ** 2))
    rv.append(df.groupBy(df.a).agg(F.max(df.b**2)))
    rv.append(df.groupBy(df.a).agg(F.max(df.b + df.a)))

    rv.append(df.groupBy(df.a).agg(F.countDistinct(df.b)))

    return rv


def _test_groupBy_2(spark):
    F = (
        pspl.sql.functions
        if isinstance(spark, pspl.sql.SparkSession)
        else ps.sql.functions
    )

    df = create_test_df_2(spark)
    rv = []
    rv.append(df.groupBy(df.a).sum())
    rv.append(df.groupBy(df.a).sum("b"))
    rv.append(df.groupBy(df.a).sum("b", "a"))

    rv.append(df.groupBy(df.a).count())

    rv.append(df.groupBy(df.a).avg())
    rv.append(df.groupBy(df.a).avg("b"))
    rv.append(df.groupBy(df.a).avg("b", "a"))

    rv.append(df.groupBy(df.a, df.b).mean("c"))

    rv.append(df.groupBy(df.a).min())
    rv.append(df.groupBy(df.a).min("b"))
    rv.append(df.groupBy(df.a).min("b", "a"))
    rv.append(df.groupBy(df.a, df.b).min("c"))

    rv.append(df.groupBy(df.a).max())
    rv.append(df.groupBy(df.a).max("b"))
    rv.append(df.groupBy(df.a).max("b", "a"))
    rv.append(df.groupBy(df.a, df.b).max("c"))

    rv.append(df.groupBy(df.a**2).agg(F.sum(df.b)))
    rv.append(df.groupBy(df.a**2).agg(F.sum(df.b), F.sum(df.c)))

    rv.append(df.groupBy(df.a, df.b).agg(F.sum(df.c)))
    rv.append(df.groupBy(df.a, df.b).agg(F.sum(df.c), F.sum(df.b)))

    return rv


def _test_withColumn_withColumns(spark):
    df = create_test_df_1(spark)
    rv = []
    rv.append(df.withColumn("a1", df.a))
    rv.append(df.withColumn("a1", df.a + 1))
    rv.append(df.withColumn("a1", df.a + df.b))
    rv.append(df.withColumn("a1", df.a + df.b + 1))
    rv.append(df.withColumn("a1", df.a + df.b + df.a))
    rv.append(df.withColumn("a1", df.a + df.b - 1))
    rv.append(df.withColumn("a1", df.a + df.b - df.a))
    rv.append(df.withColumn("a1", df.a + df.b * 1))
    rv.append(df.withColumn("a1", df.a + df.b * df.a))
    rv.append(df.withColumn("a1", df.a + df.b / 1))
    rv.append(df.withColumn("a1", df.a + df.b / df.a))

    rv.append(df.withColumn("a1", df.a - df.b))
    rv.append(df.withColumn("a1", df.a - df.b - 1))
    rv.append(df.withColumn("a1", df.a - df.b - df.a))
    rv.append(df.withColumn("a1", df.a - df.b * 1))
    rv.append(df.withColumn("a1", df.a - df.b * df.a))
    rv.append(df.withColumn("a1", df.a - df.b / 1))
    rv.append(df.withColumn("a1", df.a - df.b / df.a))

    rv.append(df.withColumn("a1", df.a * df.b))
    rv.append(df.withColumn("a1", df.a * df.b * 1))
    rv.append(df.withColumn("a1", df.a * df.b * df.a))
    rv.append(df.withColumn("a1", df.a * df.b + 1))
    rv.append(df.withColumn("a1", df.a * df.b + df.a))
    rv.append(df.withColumn("a1", df.a * df.b - 1))
    rv.append(df.withColumn("a1", df.a * df.b - df.a))
    rv.append(df.withColumn("a1", df.a * df.b / 1))

    rv.append(df.withColumns({"a1": df.a, "b1": df.b}))
    rv.append(df.withColumns({"a1": df.a + 1, "b1": df.b + 1}))
    rv.append(df.withColumns({"a1": df.a + df.b, "b1": df.b + df.a}))
    rv.append(df.withColumns({"a1": df.a + df.b + 1, "b1": df.b + df.a + 1}))
    rv.append(df.withColumns({"a1": df.a + df.b + df.a, "b1": df.b + df.a}))
    rv.append(df.withColumns({"a1": df.a + df.b - 1, "b1": df.b + df.a - 1}))
    rv.append(df.withColumns({"a1": df.a + df.b - df.a, "b1": df.b + df.a}))
    rv.append(df.withColumns({"a1": df.a + df.b * 1, "b1": df.b + df.a * 1}))

    return rv


def _test_withColumnRenamed(spark):
    df = create_test_df_1(spark)
    rv = []
    rv.append(df.withColumnRenamed("a", "a1"))
    rv.append(df.withColumnRenamed("a", "a1").withColumnRenamed("b", "b1"))
    rv.append(
        df.withColumnRenamed("a", "a1")
        .withColumnRenamed("b", "b1")
        .withColumnRenamed("a1", "a2")
    )

    return rv


def _test_sort(spark):
    df = create_test_df_1(spark)
    rv = []
    rv.append(df.sort(df.a))
    rv.append(df.sort(df.a, df.b))

    rv.append(df.sort(df.a, ascending=False))
    rv.append(df.sort(df.a, df.b, ascending=False))
    rv.append(df.sort(df.a, df.b, ascending=[True, False]))

    rv.append(df.select("a").sort("a"))
    rv.append(df.select("a").sort("a", ascending=False))
    rv.append(df.select("a").sort("a", ascending=[False]))

    rv.append(df.sort(df.a.asc()))
    rv.append(df.sort(df.a.asc(), df.b.asc()))
    rv.append(df.sort(df.a.asc(), df.b.asc(), df.c.asc()))
    rv.append(df.sort(df.a.asc(), df.b.asc(), df.c.asc(), df.d.asc()))

    rv.append(df.sort(df.a.desc()))
    rv.append(df.sort(df.a.desc(), df.b.desc()))
    rv.append(df.sort(df.a.desc(), df.b.desc(), df.c.desc()))
    rv.append(df.sort(df.a.desc(), df.b.desc(), df.c.desc(), df.d.desc()))

    rv.append(df.sort(df.a.asc(), df.b.desc()))
    rv.append(df.sort(df.a.asc(), df.b.desc(), df.c.asc()))
    rv.append(df.sort(df.a.asc(), df.b.desc(), df.c.asc(), df.d.desc()))

    rv.append(df.sort(df.a * -1))
    rv.append(df.sort(df.a * -1, df.b * -1))

    rv.append(df.sort(-df.a))
    rv.append(df.sort(-df.a, -df.b))
    rv.append(df.sort(-df.a, df.b))
    rv.append(df.sort(-df.a, df.b, df.c))
    rv.append(df.sort(-df.a, df.b, df.c, df.d))
    rv.append(df.sort(-df.a, -df.b, df.c, df.d))

    rv.append(df.sort(-df.a, -df.b, df.c, df.d.desc_nulls_last()))

    return rv


def _create_test_dfs_for_join_1(spark):
    dfl = spark.createDataFrame(
        [[int(i / 10), int(i / 5), i] for i in range(100)],
        schema=["a", "b", "c"],
    )
    dfr = spark.createDataFrame(
        [[int(i / 10), int(i / 5), i] for i in range(0, 100, 2)],
        schema=["a", "d", "e"],
    )

    return dfl, dfr


def _create_test_dfs_for_join_2(spark):
    dfl = spark.createDataFrame(
        [["A", 2, 3], ["B", 5, 6], ["C", 2, 3]], ["a", "b", "c"]
    )
    dfr = spark.createDataFrame(
        [["A", 2, 3], ["B", 5, 6], ["D", 2, 3]], ["a", "d", "e"]
    )

    return dfl, dfr


def _create_test_dfs_for_join_3(spark):
    dfl = spark.createDataFrame(
        [["A", 2, 3], ["B", 5, 6], ["C", 2, 3], ["C", 2, 3]], ["a", "b", "c"]
    )
    dfr = spark.createDataFrame(
        [["A", 2, 3], ["B", 5, 6], ["D", 2, 3], ["D", 2, 3]], ["a", "d", "e"]
    )

    return dfl, dfr


def _test_join_1(spark):
    dfl, dfr = _create_test_dfs_for_join_1(spark)
    rv = []
    rv.append(dfl.join(dfr, "a"))
    [
        rv.append(dfl.join(dfr, "a" if h != "cross" else None, how=h))
        for h in [
            "inner",
            "outer",
            "full",
            "fullouter",
            "full_outer",
            "left",
            "leftouter",
            "left_outer",
            "right",
            "rightouter",
            "right_outer",
            "semi",
            "leftsemi",
            "left_semi",
            "anti",
            "leftanti",
            "left_anti",
        ]
    ]
    rv.append(dfl.join(dfr.withColumnRenamed("a", "a_left"), how="cross"))

    rv.append(dfr.join(dfl, ["a"], how="left"))

    return rv


def _test_join_2(spark):
    dfl, dfr = _create_test_dfs_for_join_2(spark)
    rv = []
    rv.append(dfl.join(dfr, "a"))
    [
        rv.append(dfl.join(dfr, "a" if h != "cross" else None, how=h))
        for h in [
            "inner",
            "outer",
            "full",
            "fullouter",
            "full_outer",
            "left",
            "leftouter",
            "left_outer",
            "right",
            "rightouter",
            "right_outer",
            "semi",
            "leftsemi",
            "left_semi",
            "anti",
            "leftanti",
            "left_anti",
        ]
    ]
    rv.append(dfl.join(dfr.withColumnRenamed("a", "a_left"), how="cross"))

    rv.append(dfr.join(dfl, ["a"], how="left"))
    # rv.append(dfr.join(dfl, dfl.a == dfr.a, how="inner"))

    return rv


def _test_join_3(spark):
    dfl, dfr = _create_test_dfs_for_join_3(spark)
    rv = []
    [
        rv.append(
            dfl.join(dfr, dfl.a == dfr.a if h != "cross" else None, how=h)
        )
        for h in [
            "inner",
            "outer",
            "right",
            "semi",
            "anti",
        ]
    ]

    return rv


def run_compare_test(dfs_generator):
    # genreates dataframes using the given generator with polars and spark
    # as engines and compares the results
    sparkpl = SessionHelper.session(
        SessionHelper.Engine.POLARS
    ).builder.getOrCreate()
    spark = SessionHelper.session(
        SessionHelper.Engine.SPARK
    ).builder.getOrCreate()
    pl_results = dfs_generator(sparkpl)
    ps_results = dfs_generator(spark)
    for i in range(len(pl_results)):
        assert compare_pl_spark(pl_results[i], ps_results[i])


def test_filter():
    run_compare_test(_test_filter)


def test_groupBy():
    run_compare_test(_test_groupBy)


def test_groupBy_2():
    run_compare_test(_test_groupBy_2)


def test_withColumn_withColumns():
    run_compare_test(_test_withColumn_withColumns)


def test_withColumnRenamed():
    run_compare_test(_test_withColumnRenamed)


def test_sort():
    run_compare_test(_test_sort)


def test_select():
    run_compare_test(_test_select)


def test_join_1():
    run_compare_test(_test_join_1)


def test_join_2():
    run_compare_test(_test_join_2)


def test_join_3():
    run_compare_test(_test_join_3)


if __name__ == "__main__":
    test_collect()
    test_join_3()
    test_join_2()
    test_join_1()
    test_groupBy_2()
    test_sort()
    test_withColumnRenamed()
    test_withColumn_withColumns()
    test_groupBy()
    test_dataframe_select()
    test_select()
    test_filter()
