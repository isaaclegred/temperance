"""
Basic tests for temperance.core.result.

TODO: extend to cover EoSPosterior and InferenceResult.
"""

import numpy as np
import pandas as pd

from core import result


def test_uniform_mass_pdf_support():
    samples = pd.DataFrame({"m1": [2.0, 1.5, 3.0], "m2": [1.2, 1.8, 1.2]})
    pdf = result.uniform_mass_pdf(samples)
    expected = 2 / (2.5 - 1.0) ** 2
    np.testing.assert_allclose(pdf, [expected, 0.0, 0.0])


def test_get_weight_columns_infers_log_and_linear():
    samples = pd.DataFrame({"eos": [0, 1], "logweight_a": [0.0, -1.0], "weight_b": [1.0, 2.0]})
    columns = {c.name: c for c in result.get_weight_columns(samples, None)}
    assert set(columns) == {"logweight_a", "weight_b"}
    assert columns["logweight_a"].is_log
    assert not columns["weight_b"].is_log


def test_get_column_weight_log_and_inverted():
    samples = pd.DataFrame({"logweight": [0.0, np.log(2.0)]})
    column = result.WeightColumn("logweight", True, False)
    np.testing.assert_allclose(result.get_column_weight(samples, column), [1.0, 2.0])
    np.testing.assert_allclose(
        result.get_column_weight(samples, column.get_inverse()), [1.0, 0.5]
    )
