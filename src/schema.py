import pandera.pandas as pa
from pandera.typing import Series


class FundamentalsSchema(pa.DataFrameModel):
    gvkey: Series[str]
    datadate: Series[pa.DateTime]
    fyear: Series[int] = pa.Field(ge=2010, le=2024)
    tic: Series[str]
    at: Series[float] = pa.Field(ge=0, nullable=True)
    lt: Series[float] = pa.Field(ge=0, nullable=True)
    revt: Series[float] = pa.Field(nullable=True)
    ni: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
        coerce = True
        unique = ["gvkey", "fyear"]


class MdnaSchema(pa.DataFrameModel):
    gvkey: Series[str]
    fyear: Series[int] = pa.Field(ge=2010, le=2024)
    cik: Series[str]
    fdate: Series[pa.DateTime]
    mdna_text: Series[str] = pa.Field(str_length={"min_value": 5_000, "max_value": 300_000})
    char_count: Series[int] = pa.Field(ge=5_000, le=300_000)

    class Config:
        strict = "filter"
        coerce = True
        unique = ["gvkey", "fyear"]
