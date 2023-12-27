from datetime import date
import pyspark as ps
import pyspark.sql.functions as psfunctions
import pysparkpl.sql.types as types


def test_typename_to_dtype():
    spark = ps.sql.SparkSession.builder.getOrCreate()
    ps_df = spark.createDataFrame(
        [
            [1, 1, 1.0, 1.0, "string1", date(2023, 12, 27)],
            [2, 2, 2.0, 2.0, "string2", date(2023, 12, 27)],
        ],
        schema=types.StructType(
            [
                types.StructField("a", types.IntegerType()),
                types.StructField("b", types.LongType()),
                types.StructField("c", types.FloatType()),
                types.StructField("d", types.DoubleType()),
                types.StructField("e", types.StringType()),
                types.StructField("f", types.DateType()),
            ]
        ),
    )
    for tn in [
        "int",
        "integer",
        "short",
        "byte",
        "long",
        "bigint",
        "float",
        "double",
        "string",
        "boolean",
        "binary",
    ]:
        casted = ps_df.select(psfunctions.col("a").astype(tn))
        assert casted.schema.fields[0].dataType == types._type_name_to_dtype[tn]

    casted = ps_df.select(psfunctions.col("f").astype("date"))
    assert casted.schema.fields[0].dataType == types._type_name_to_dtype["date"]


if __name__ == "__main__":
    test_typename_to_dtype()
