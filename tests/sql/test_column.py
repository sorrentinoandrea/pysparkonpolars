import polars as pl
from pysparkpl.sql.column import Column


def test_column_1():
    assert Column("a")._name == "a"
    assert Column("a").alias("b")._name == "b"
    assert Column("a").asc()._name == "a ASC NULLS FIRST"
    assert Column("a").desc()._name == "a DESC NULLS LAST"
    assert Column("a").asc_nulls_first()._name == "a ASC NULLS FIRST"
    assert Column("a").desc_nulls_first()._name == "a DESC NULLS FISRT"
    assert Column("a").asc_nulls_last()._name == "a ASC NULLS LAST"

    assert (Column("a") + 1)._name == "(a + 1)"
    assert (Column("a") + Column("b"))._name == "(a + b)"
    assert (Column("a") + Column("b") + 1)._name == "((a + b) + 1)"

    assert (Column("a") - 1)._name == "(a - 1)"
    assert (Column("a") - Column("b"))._name == "(a - b)"
    assert (Column("a") - Column("b") - 1)._name == "((a - b) - 1)"

    assert (Column("a") * 1)._name == "(a * 1)"
    assert (Column("a") * Column("b"))._name == "(a * b)"
    assert (Column("a") * Column("b") * 1)._name == "((a * b) * 1)"

    assert (Column("a") / 1)._name == "(a / 1)"
    assert (Column("a") / Column("b"))._name == "(a / b)"
    assert (Column("a") / Column("b") / 1)._name == "((a / b) / 1)"

    assert (Column("a").astype("int"))._name == "CAST(a as INT)"
    assert (Column("a").astype("float"))._name == "CAST(a as FLOAT)"
    assert (Column("a").astype("string"))._name == "CAST(a as STRING)"
    assert (Column("a").astype("boolean"))._name == "CAST(a as BOOLEAN)"
    assert (Column("a").astype("date"))._name == "CAST(a as DATE)"


if __name__ == "__main__":
    test_column_1()
