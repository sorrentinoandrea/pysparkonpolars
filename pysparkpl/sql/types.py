from typing import Union
import polars as pl
from pyspark.sql.types import (
    AtomicType,
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
    ArrayType,
    MapType,
    BooleanType,
    DateType,
    TimestampType,
    NullType,
    BinaryType,
    DataType,
    DecimalType,
    DoubleType,
    LongType,
    ShortType,
    ByteType,
    UserDefinedType,
)

type_mapping = {
    StringType(): pl.Utf8,
    IntegerType(): pl.Int32,
    LongType(): pl.Int64,
    FloatType(): pl.Float32,
    DoubleType(): pl.Float64,
    BooleanType(): pl.Boolean,
    DateType(): pl.Date,
    NullType(): pl.Null,
    BinaryType(): pl.Binary,
    ShortType(): pl.Int16,
    ByteType(): pl.Int8,
    ArrayType(StringType()): pl.List(pl.Utf8),
    ArrayType(IntegerType()): pl.List(pl.Int32),
    ArrayType(LongType()): pl.List(pl.Int64),
    ArrayType(FloatType()): pl.List(pl.Float32),
    ArrayType(DoubleType()): pl.List(pl.Float64),
    ArrayType(BooleanType()): pl.List(pl.Boolean),
    ArrayType(DateType()): pl.List(pl.Date),
    ArrayType(NullType()): pl.List(pl.Null),
    ArrayType(BinaryType()): pl.List(pl.Binary),
    ArrayType(ShortType()): pl.List(pl.Int16),
    ArrayType(ByteType()): pl.List(pl.Int8),
}

_type_name_to_dtype = {
    "string": StringType(),
    "int": IntegerType(),
    "integer": IntegerType(),
    "long": LongType(),
    "bigint": LongType(),
    "float": FloatType(),
    "double": DoubleType(),
    "boolean": BooleanType(),
    "date": DateType(),
    "null": NullType(),
    "binary": BinaryType(),
    "short": ShortType(),
    "byte": ByteType(),
}


def map_type(
    t: DataType,
) -> Union[pl.datatypes.classes.DataTypeClass, pl.Struct]:
    if isinstance(t, UserDefinedType):
        raise NotImplementedError("UserDefinedType is not supported")
    elif isinstance(t, DecimalType):
        raise NotImplementedError("DecimalType is not supported")
    elif isinstance(t, MapType):
        raise NotImplementedError("MapType is not supported")
    elif isinstance(t, StructType):
        return pl.Struct(
            [pl.Field(f.name, map_type(f.dataType)) for f in t.fields]
        )
    elif t in type_mapping:
        return type_mapping[t]
    else:
        raise NotImplementedError(f"Type {t} is not supported")
