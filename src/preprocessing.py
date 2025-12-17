import re
from pandas.errors import ParserError
import pandas as pd
import numpy as np
from typing import Iterable, Optional


def read_raw_file(path: str) -> pd.DataFrame:
    """
    Read a raw spectrometer CSV file (skip first header row) into a DataFrame.

    Tries a normal comma/semicolon separated read first. On parse error attempts a
    tab-separated read and normalizes decimal commas to dots.

    Parameters
    ----------
    path :
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame (including original column names).
    """
    try:
        return pd.read_csv(path, skiprows=1)
    except ParserError:
        # fallback for files with tabs or comma-as-decimal
        return (
            pd.read_csv(path, skiprows=1, sep="\t", engine="python")
            .astype("str")
            .replace(",", ".", regex=True)
            .astype("float")
        )


def select_expected_columns(df: pd.DataFrame, expected_keys: Iterable) -> pd.DataFrame:
    """
    Select only the columns that are present in expected_keys.

    If some expected keys are missing, prints the filename (caller responsibility)
    and returns the available subset (preserves order of expected_keys when possible).

    Parameters
    ----------
    df :
        Input DataFrame.
    expected_keys :
        Iterable of expected column keys (e.g. elements_dict.keys()).

    Returns
    -------
    pd.DataFrame
    """
    keys = list(expected_keys)
    try:
        return df[keys]
    except KeyError:
        # preserve behavior from original: notify and return what we can
        missing = set(keys) - set(df.columns)
        print(f"Missing expected columns: {', '.join(sorted(missing))}")
        # return intersection in original order
        available = [k for k in keys if k in df.columns]
        return df[available]

def despike(data):

    data_despike = np.zeros(len(data)-2)
    for i in range(1, len(data)-2):
        if data[i] > np.mean([data[i-1], data[i+1]]): 
            data_despike[i] = np.mean([data[i-1], data[i+1]])
        else:
            data_despike[i] = data[i]
    return data_despike


def round_and_despike_df(df: pd.DataFrame):
    """
    Round values to integers, apply despike per-column, drop NA and reset index.

    Note: this function assumes a despike function is available in the module scope.
    """
    df = df.round(0)
    df = df.apply(despike, axis=0)
    df = df.dropna().reset_index(drop=True)
    return df


def normalize_by_elements_dict(df: pd.DataFrame, elements_dict: dict) -> pd.DataFrame:
    """
    Normalize columns by their expected isotope values from elements_dict.

    Appends a final row with elements_dict values, divides each column by that row,
    then removes the helper row and strips digits from column names.

    Parameters
    ----------
    df :
        Input DataFrame with columns matching elements_dict keys.
    elements_dict :
        Mapping from column name to isotope value used for normalization.

    Returns
    -------
    pd.DataFrame
    """
    # insert mapping row at the end and normalize
    df.loc[-1] = df.columns.map(elements_dict)
    df_normalized = df.apply(lambda x: x / x.iloc[-1])
    df_normalized = df_normalized.iloc[:-1, :]
    df_normalized.columns = [re.sub(r"\d+", "", x) for x in df_normalized.columns]
    df_normalized = df_normalized.round(0)
    return df_normalized


def subtract_blank_and_clip(df: pd.DataFrame, n_std: float = 1.0, row_nr = 31) -> np.ndarray:
    """
    Compute an instrument blank from rows 1:row_nr and subtract (mean + n_std*std),
    then clip values (preserves original behavior: clip lower bound 1).

    Parameters
    ----------
    arr :
        2D numpy array of values.
    n_std :
        Number of standard deviations to add to the mean when computing blank.
    row_nr :
        At which row the instrument blank ends.

    Returns
    -------
    np.ndarray
    """
    arr = df.values
    means = np.mean(arr[1:row_nr, :], axis=0) + n_std * np.std(arr[1:row_nr, :], axis=0)
    res = (arr - means).clip(1)
    normalized_blank = pd.DataFrame(data = res, columns=df.columns)
    return normalized_blank


def apply_multiple_despike(df: pd.DataFrame, n_des: int = 1):
    """
    Apply despike (per-column) n_des times to a numpy array.

    Requires despike available in module scope.
    """
    res = df.values
    for _ in range(n_des):
        res = np.apply_along_axis(despike, 0, res)
    despike_df = pd.DataFrame(data=res, columns=df.columns)
    return despike_df


def sort_and_bunch(df: pd.DataFrame, n: int = 10, bunch_no: int = 10) -> pd.DataFrame:
    """
    Sort by 'Fe' descending, remove first n rows (outlier bunch) and compute means
    for successive groups of n rows (bunching).

    Parameters
    ----------
    df :
        Input DataFrame with column 'Fe'.
    n :
        Bunch size.
    bunch_no :
        Number of bunches to keep (groups of 9 rows each) after dropping first bunch.

    Returns
    -------
    pd.DataFrame
    """
    file_sorted = df.sort_values(by="Fe", ascending=False).reset_index(drop=True)
    to_bunch = file_sorted.iloc[n: ((1 + bunch_no) * n), :]
    bunched = to_bunch.groupby(to_bunch.index // n).mean().round(5)
    return bunched


def normalize_to_fe(bunched: pd.DataFrame, keep_fe: bool = True) -> pd.DataFrame:
    """
    Optionally normalize all columns to Fe (divide each column by Fe for that row).
    Keeps original Fe column if keep_fe is True.

    Parameters
    ----------
    bunched :
        Bunched DataFrame containing 'Fe' column.
    keep_fe :
        If True, restore original Fe values after normalization.
    """
    fe = bunched["Fe"].copy()
    bunched = bunched.apply(lambda x: x / bunched["Fe"]).round(5)
    if keep_fe:
        bunched["Fe"] = fe
    return bunched


def add_names_column(bunched: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Add a 'names' column formed from the filename (path) and the row index.

    Parameters
    ----------
    bunched :
        Bunched DataFrame.
    path :
        Original file path used to derive filename part.
    """
    filename = path[path.rfind("/") + 1 : -3]
    bunched["names"] = [f"{filename}_{i}" for i in range(len(bunched))]
    return bunched


def machine(path: str, elements_dict: dict, n_std: int = 1, row_nr: int = 31, n_des: int = 1, n : int = 10, bunch_no: int = 10, to_fe: bool = False, keep_fe: bool = True):
    """
    High-level preprocessing pipeline for a raw spectrometer file.

    Steps:
    1. Read raw file (skip first header row).
    2. Select only expected columns (elements_dict keys).
    3. Round and despike raw measurements.
    4. Normalize by isotope values (elements_dict).
    5. Subtract instrument blank (mean + n_std*std) and clip.
    6. Optionally apply additional despike passes.
    7. Sort by Fe and compute bunched means (groups of n rows), dropping the first bunch.
    8. Optionally normalize to Fe.
    9. Add indexed sample names and return the bunched DataFrame.

    Parameters
    ----------
    data :
        Path to the raw input file.
    n_std :
        Number of standard deviations when computing the blank.
    n_des :
        Number of despike passes to apply after blank subtraction.
    to_fe :
        If True, normalize final bunched values to Fe.
    bunch_no :
        Number of bunches (groups of 9 rows) to produce.

    Returns
    -------
    pd.DataFrame
        Final bunched DataFrame ready for downstream processing.

    Notes
    -----
    This function relies on two names available in module scope:
    - elements_dict : mapping of expected raw column names to isotope values
    - despike : function that accepts a 1D array/Series and returns a despiked 1D array/Series
    """
    # load raw
    raw_df = read_raw_file(path)

    # ensure expected columns selected (elements_dict should be defined in module or imported)
    selected = select_expected_columns(raw_df, elements_dict)

    # initial rounding and despike
    cleaned = round_and_despike_df(selected)

    # normalize by elements_dict mapping
    normalized = normalize_by_elements_dict(cleaned, elements_dict)

    # subtract blank and clip
    normalized_blank = subtract_blank_and_clip(normalized, n_std=n_std, row_nr=row_nr)

    # optional additional despike passes
    despike_df = apply_multiple_despike(normalized_blank, n_des=n_des)

    # sort and create bunches
    bunched = sort_and_bunch(despike_df, n=n, bunch_no=bunch_no)

    # optional normalize to Fe
    if to_fe:
        bunched = normalize_to_fe(bunched, keep_fe=keep_fe)

    # add names column and return
    bunched = add_names_column(bunched, path)
    return bunched
