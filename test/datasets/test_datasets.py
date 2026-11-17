import re

import numpy as np
import pandas as pd
import pytest

from temperance import datasets


TABLES = ["gravitational_waves", "nicer_mass_radius", "pulsar_masses"]


def test_tables_are_listed():
    assert datasets.list_tables() == TABLES


@pytest.mark.parametrize("table", TABLES)
def test_table_loads_and_matches_dictionary(table):
    frame = datasets.load_table(table)
    assert len(frame) > 0
    assert set(frame.columns) == set(datasets.column_dictionary(table)["column"])
    floats = datasets.column_dictionary(table).query("type == 'float'")["column"]
    for column in floats:
        assert pd.api.types.is_float_dtype(frame[column]), column


@pytest.mark.parametrize("table", TABLES)
def test_identifiers_are_well_formed(table):
    frame = datasets.load_table(table)
    for arxiv in frame["arxiv"][frame["arxiv"] != ""]:
        assert re.fullmatch(r"\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7}", arxiv), arxiv
    for doi in frame["doi"][frame["doi"] != ""]:
        assert re.fullmatch(r"10\.\d{4,9}/\S+", doi), doi
    assert (frame["analysis"].str.fullmatch(r"[a-z0-9_]+")).all()
    assert ((frame["credible_level"] > 0) & (frame["credible_level"] < 1)).all()


@pytest.mark.parametrize("table, quantity", [
    ("pulsar_masses", "mass"),
    ("nicer_mass_radius", "mass"),
    ("nicer_mass_radius", "radius"),
    ("gravitational_waves", "chirp_mass_source"),
    ("gravitational_waves", "mass_1_source"),
    ("gravitational_waves", "mass_2_source"),
    ("gravitational_waves", "lambda_tilde"),
    ("gravitational_waves", "radius_1"),
])
def test_intervals_are_ordered(table, quantity):
    frame = datasets.load_table(table)
    lower, upper = datasets.credible_interval(frame, quantity)
    given = lower.notna()
    assert given.any()
    assert (upper[given] > lower[given]).all()
    median = frame[quantity][given].dropna()
    assert ((lower[median.index] <= median) & (median <= upper[median.index])).all()


def test_values_are_plausible():
    pulsars = datasets.load_table("pulsar_masses")
    # loose bounds: some Shapiro-delay-only masses (agazie2023) are far from 1.4
    assert pulsars["mass"].dropna().between(0.5, 5.0).all()
    nicer = datasets.load_table("nicer_mass_radius")
    assert nicer["radius"].between(8.0, 16.0).all()
    assert nicer["radio_mass_prior"].dtype == "boolean"
    gw = datasets.load_table("gravitational_waves")
    lower, upper = datasets.credible_interval(gw, "mass_2_source")
    assert lower.between(0.9, 3.0).all() and upper.between(0.9, 3.0).all()


def test_every_pulsar_row_constrains_a_mass():
    # a mass, or a mass function with the total mass or the mass ratio
    pulsars = datasets.load_table("pulsar_masses")
    direct = pulsars["mass"].notna()
    binary = pulsars["mass_function"].notna() & (pulsars["total_mass"].notna() | pulsars["mass_ratio"].notna())
    assert (direct ^ binary).all()
    assert set(pulsars["component"]) == {"neutron_star", "companion"}


def test_fan2024_population_sample():
    # Table I of Fan et al. 2024 has 136 masses; the 7 from GW events are in gravitational_waves
    pulsars = datasets.load_table("pulsar_masses")
    fan = pulsars[pulsars["compilations"].str.split().apply(lambda tags: "fan2024" in tags)]
    assert len(fan) == 129
    assert (fan["mass"].notna().sum(), fan["total_mass"].notna().sum(), fan["mass_ratio"].notna().sum()) \
        == (106, 17, 6)


def test_credible_interval_uses_either_form():
    frame = pd.DataFrame({"q": [1.0, np.nan], "q_minus": [0.1, np.nan], "q_plus": [0.2, np.nan],
                          "q_lower": [np.nan, 3.0], "q_upper": [np.nan, 4.0]})
    lower, upper = datasets.credible_interval(frame, "q")
    np.testing.assert_allclose(lower, [0.9, 3.0])
    np.testing.assert_allclose(upper, [1.2, 4.0])


# --- a table directory of our own, to check that new tables work and bad ones are refused

DICTIONARY = """table,column,type,unit,required,key,description
*,name,str,,true,true,name
*,analysis,str,,true,true,analysis
*,samples_url,url,,false,false,samples
example,x,float,km,true,false,a quantity
example,x_minus,float,km,true,false,lower offset
example,x_plus,float,km,true,false,upper offset
example,flag,bool,,false,false,a flag
"""


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    (tmp_path / "columns.csv").write_text(DICTIONARY)
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    return tmp_path


def test_new_table_needs_only_a_csv_and_dictionary_rows(data_dir):
    (data_dir / "example.csv").write_text(
        "name,analysis,x,x_minus,x_plus,flag,samples_url\n"
        "A,one,1.0,0.1,0.2,true,https://example.org/a\n"
        "A,two,1.1,0.1,0.1,,\n")
    assert datasets.list_tables() == ["example"]
    frame = datasets.load_table("example")
    assert list(frame["analysis"]) == ["one", "two"]
    assert frame["flag"].dtype == "boolean"
    assert bool(frame["flag"].iloc[0]) and frame["flag"].isna().iloc[1]


@pytest.mark.parametrize("rows, message", [
    ("A,one,1.0,0.1,0.2,,\nA,one,1.1,0.1,0.1,,\n", "repeated"),
    ("A,,1.0,0.1,0.2,,\n", "required"),
    ("A,one,abc,0.1,0.2,,\n", "column x:"),
    ("A,one,1.0,0.1,0.2,maybe,\n", "true or false"),
    ("A,one,1.0,0.1,0.2,,http://insecure\n", "https"),
])
def test_bad_tables_are_refused(data_dir, rows, message):
    (data_dir / "example.csv").write_text("name,analysis,x,x_minus,x_plus,flag,samples_url\n" + rows)
    with pytest.raises(ValueError, match=message):
        datasets.load_table("example")


def test_columns_must_match_dictionary(data_dir):
    (data_dir / "example.csv").write_text("name,analysis,x,x_minus,x_plus,flag,samples_url,y\n"
                                          "A,one,1,0.1,0.1,,,2\n")
    with pytest.raises(ValueError, match="undeclared"):
        datasets.load_table("example")
    with pytest.raises(ValueError, match="unknown table"):
        datasets.load_table("missing")


def test_key_columns_come_from_the_dictionary(data_dir):
    # with x in the key, two rows may share name and analysis
    (data_dir / "columns.csv").write_text(DICTIONARY.replace("example,x,float,km,true,false", "example,x,float,km,true,true"))
    (data_dir / "example.csv").write_text("name,analysis,x,x_minus,x_plus,flag,samples_url\n"
                                          "A,one,1.0,0.1,0.2,,\nA,one,2.0,0.1,0.2,,\n")
    assert len(datasets.load_table("example")) == 2


def test_incomplete_intervals_are_refused(data_dir):
    dictionary = DICTIONARY.replace("example,x_minus,float,km,true,", "example,x_minus,float,km,false,")
    (data_dir / "columns.csv").write_text(dictionary)
    (data_dir / "example.csv").write_text("name,analysis,x,x_minus,x_plus,flag,samples_url\n"
                                          "A,one,1.0,,0.2,,\n")
    with pytest.raises(ValueError, match="incomplete"):
        datasets.load_table("example")
