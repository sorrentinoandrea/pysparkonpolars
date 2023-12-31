import polars as pl


class GroupedData(object):
    """
    A set of methods for aggregations on a :class:`DataFrame`, created by
    :func:`DataFrame.
    """

    def __init__(self, group_by: pl.GroupBy):
        self._group_by = group_by

    def sum(self, *cols: Union[str, col]):
        return DataFrame(self._group_by.sum(*cols))
