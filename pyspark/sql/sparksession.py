
from typing import Union, Optional, Iterable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql.pandas._typing import DataFrameLike as PandasDataFrameLike, ArrayLike

class SparkSession:

    def __init__(self):
        pass

    def createDataFrame(self, data: Union[Iterable[Any], "PandasDataFrameLike", "ArrayLike"] , schema, samplingRatio, verifySchema):
        pass