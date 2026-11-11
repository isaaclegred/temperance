"""
Published measurements useful for EoS inference, as CSV tables.

Each table has one row per analysis, so a pulsar or event analysed several
times appears in several rows, told apart by the ``analysis`` column.  The
column dictionary ``columns.csv`` declares every table's columns with their
types and units; a new kind of data is a new ``<table>.csv`` plus its rows in
the dictionary (see README.md in this directory).

list_tables            the names of the tables
column_dictionary      the declared columns of a table
load_table             a table as a DataFrame, checked against the dictionary
credible_interval      the lower and upper ends of a quantity's interval
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

# columns.csv lists the columns every table shares under this table name
COMMON = "*"


def _read(path):
    # read everything as text, so blanks stay blank and nothing is guessed;
    # types come from the dictionary
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def list_tables():
    """The names of the tables declared in columns.csv."""
    tables = set(_read(DATA_DIR / "columns.csv")["table"])
    return sorted(tables - {COMMON})


def column_dictionary(table=None):
    """
    The rows of columns.csv (table, column, type, unit, required, key,
    description): the shared columns and those of ``table``, or the whole
    dictionary if table is None.  The columns marked key identify a row.
    """
    dictionary = _read(DATA_DIR / "columns.csv")
    if table is None:
        return dictionary
    if table not in list_tables():
        raise ValueError(f"unknown table {table!r}; the tables are {list_tables()}")
    return dictionary[dictionary["table"].isin([COMMON, table])].reset_index(drop=True)


def _convert(values, kind, where):
    blank = values == ""
    if kind == "float":
        try:
            return pd.to_numeric(values.mask(blank, np.nan))
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
    if kind == "bool":
        unknown = ~values.isin(["true", "false", ""])
        if unknown.any():
            raise ValueError(f"{where}: expected true or false, got {sorted(set(values[unknown]))}")
        return values.map({"true": True, "false": False, "": pd.NA}).astype("boolean")
    if kind == "url":
        bad = ~blank & ~values.str.startswith("https://")
        if bad.any():
            raise ValueError(f"{where}: not an https URL: {sorted(set(values[bad]))}")
        return values
    if kind == "str":
        return values
    raise ValueError(f"{where}: unknown type {kind!r} in columns.csv")


def _check_intervals(frame, table):
    # a quantity is given either as a median with offsets (q, q_minus,
    # q_plus) or as the ends of an interval (q_lower, q_upper), and each
    # form must be complete or absent
    for column in frame.columns:
        forms = [[column, f"{column}_minus", f"{column}_plus"], [f"{column}_lower", f"{column}_upper"]]
        forms = [form for form in forms if all(c in frame for c in form)]
        filled = [frame[form].notna() for form in forms]
        for form, given in zip(forms, filled):
            partial = given.any(axis=1) & ~given.all(axis=1)
            if partial.any():
                raise ValueError(f"{table}.csv: incomplete {form} in rows {list(frame.index[partial])}")
        if len(filled) == 2:
            both = filled[0].all(axis=1) & filled[1].all(axis=1)
            if both.any():
                raise ValueError(f"{table}.csv: {column} given both as median and as bounds "
                                 f"in rows {list(frame.index[both])}")


def load_table(table):
    """
    Read ``<table>.csv`` and check it against the column dictionary: every
    declared column is present and no other, required and key columns are
    filled, values have the declared types, the key columns identify each
    row uniquely, and interval columns are filled consistently.

    Returns
    -------
    pandas.DataFrame, one row per analysis.  Float columns are NaN, and
    text columns empty, where nothing is reported.
    """
    columns = column_dictionary(table)
    frame = _read(DATA_DIR / f"{table}.csv")
    declared = list(columns["column"])
    missing = [c for c in declared if c not in frame]
    extra = [c for c in frame if c not in declared]
    if missing or extra:
        raise ValueError(f"{table}.csv does not match columns.csv: missing {missing}, undeclared {extra}")
    for column in columns.itertuples():
        where = f"{table}.csv, column {column.column}"
        if "true" in (column.required, column.key) and (frame[column.column] == "").any():
            raise ValueError(f"{where}: required but blank in rows "
                             f"{list(frame.index[frame[column.column] == ''])}")
        frame[column.column] = _convert(frame[column.column], column.type, where)
    key = list(columns.loc[columns["key"] == "true", "column"])
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        raise ValueError(f"{table}.csv: ({', '.join(key)}) repeated: "
                         f"{frame.loc[duplicated, key].values.tolist()}")
    _check_intervals(frame, table)
    return frame


def credible_interval(frame, quantity):
    """
    The lower and upper ends of the credible interval of ``quantity`` in each
    row of a table, whichever way the row gives it: ``quantity_lower`` and
    ``quantity_upper``, or ``quantity`` -/+ ``quantity_minus`` /
    ``quantity_plus``.  NaN where the row reports neither.

    Returns
    -------
    (lower, upper), two pandas Series indexed like frame.
    """
    lower = frame[quantity] - frame[f"{quantity}_minus"]
    upper = frame[quantity] + frame[f"{quantity}_plus"]
    if f"{quantity}_lower" in frame:
        lower = lower.fillna(frame[f"{quantity}_lower"])
        upper = upper.fillna(frame[f"{quantity}_upper"])
    return lower, upper
