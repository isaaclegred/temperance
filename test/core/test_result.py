"""
Basic tests for temperance.core.result.

TODO: extend to cover EoSPosterior and InferenceResult.
"""

import numpy as np
import pandas as pd
import pytest

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


def make_posterior():
    samples = pd.DataFrame({"eos": np.arange(4), "logweight_a": np.log([1.0, 2.0, 3.0, 4.0]),
                            "logweight_b": np.log([0.5, 0.5, 1.0, 1.0])})
    return result.EoSPosterior(samples)


def test_explicit_weight_columns():
    samples = make_posterior().samples
    columns = [result.WeightColumn("logweight_a")]
    assert result.EoSPosterior(samples, weight_columns=columns).weight_columns_available == columns
    from_tuples = result.EoSPosterior(samples, weight_columns=[("logweight_a", True)])
    assert from_tuples.weight_columns_available == columns


def test_bad_weight_columns_raise():
    with pytest.raises(TypeError):
        result.get_weight_columns(make_posterior().samples, ["logweight_a"])


def test_posterior_get_column_weight():
    posterior = make_posterior()
    np.testing.assert_allclose(posterior.get_column_weight(result.WeightColumn("logweight_a")),
                               [1.0, 2.0, 3.0, 4.0])


def test_posterior_get_column_weight_missing():
    with pytest.raises(ValueError, match="not present"):
        make_posterior().get_column_weight(result.WeightColumn("logweight_missing"))


def test_estimate_evidence_is_silent(capsys):
    evidence, _ = make_posterior().estimate_evidence([result.WeightColumn("logweight_a")])
    assert evidence == pytest.approx(2.5)
    assert capsys.readouterr().out == ""


def test_logical_or_weight_column_naming_and_inputs_unchanged(capsys):
    posterior = make_posterior()
    columns = [result.WeightColumn("logweight_a"), result.WeightColumn("logweight_b")]
    posterior.add_logical_or_weight_column(columns)
    assert "weight_a_or_b" in posterior.samples
    assert not any(column.is_inverted for column in columns)
    _, column = result.get_logical_or_weight_column(posterior.samples, columns)
    assert column.name == "weight_a_or_b"
    assert not any(column.is_inverted for column in columns)
    assert capsys.readouterr().out == ""


def test_properties_module_imports():
    import importlib
    importlib.import_module("core.properties")
