"""The landing data contract keeps column types stable when a batch contains NULLs."""

import pandas as pd
import pyarrow as pa
import pytest

extract = pytest.importorskip("extract")


def test_integer_columns_stay_integer_with_nulls():
    df = pd.DataFrame({"video_id": [1, 2], "nasmo_seq": [123.0, None], "seq_no": [1, 2],
                       "ball_flight_code": [1.0, None], "hole": [None, None], "width": [1280.0, 1280.0],
                       "club_conf": [None, None], "carry_m": [210.0, None], "fps": [60.0, 30.0],
                       "club": ["DRIVER", None]})
    t = extract.to_arrow(df, "videos")
    assert t.schema.field("nasmo_seq").type == pa.int64()
    assert t.schema.field("hole").type == pa.int64()          # all NULL, still an integer column
    assert t.schema.field("club_conf").type == pa.float64()   # all NULL, still a float column
    assert t.schema.field("club").type in (pa.string(), pa.large_string())  # both map to varchar
    assert t.column("nasmo_seq").null_count == 1
