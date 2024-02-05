from typing import Any, List

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    age: List[Any | None]
    heart_rate: List[Any | None]
    saturation: List[Any | None]
    systolic_bp: List[Any | None]
    comm_month: List[Any | None]
    comm_type: List[Any | None]
    comm_year: List[Any | None]
    cons_conf_idx: List[Any | None] = Field(alias="cons.conf.idx")
    cons_price_idx: List[Any | None] = Field(alias="cons.price.idx")
    curr_n_contact: List[Any | None]
    curr_outcome: List[Any | None]
    customer_id: List[Any | None]
    date: List[Any | None]
    days_since_last_campaign: List[Any | None]
    default: List[Any | None]
    education: List[Any | None]
    email: List[Any | None]
    emp_var_rate: List[Any | None] = Field(alias="emp.var.rate")
    euribor3m: List[Any | None]
    first_name: List[Any | None]
    housing: List[Any | None]
    id: List[Any | None]  # noqa
    job: List[Any | None]
    last_n_contact: List[Any | None]
    last_name: List[Any | None]
    last_outcome: List[Any | None]
    loan: List[Any | None]
    marital: List[Any | None]
    nr_employed: List[Any | None] = Field(alias="nr.employed")
    phone: List[Any | None]

    @model_validator(mode="after")
    def check_all_same_length(self) -> "PredictionRequest":
        lengths = set()
        for field_name in self.model_fields:
            n_elems = len(getattr(self, field_name))
            lengths.add(n_elems)

        if len(lengths) != 1:
            err_mesg = "Not all elements are the same length"
            raise ValueError(err_mesg)
        return self

    def as_dataframe(self) -> pd.DataFrame:
        # Implementation detail: since our dataframe expects variables
        # with "." (dots) in some of the columns and that names
        # with dots are not valid Python variables, we need to set
        # `by_alias=True` so that the output JSON has keys with the dots
        dump = self.model_dump(by_alias=True)
        dataframe = pd.DataFrame(dump)
        return dataframe