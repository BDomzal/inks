
### ORIGINAL PREPROCESSING METHOD BY ALEKSANDRA TOWAREK, BARBARA WAGNER

import re
from pandas.errors import ParserError
import pandas as pd
import numpy as np
from typing import Iterable
import os

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

def despike(data: Iterable[float]) -> np.ndarray:
    """Remove single-point spikes from a 1D sequence.

    This uses a tiny 3-point window: for each interior point (indexes
    1..len(data)-3) the value is replaced by the mean of its neighbours if it
    is larger than that mean. The returned array intentionally omits the first
    and last element so its length is ``len(data) - 2`` (matching the original
    behaviour in the repository).

    Parameters
    ----------
    data:
        1D iterable (numpy array or pandas Series) of numeric values.

    Returns
    -------
    np.ndarray
        Despiked 1D array with length ``len(data) - 2``.
    """

    data_despike = np.zeros(len(data) - 2)
    for i in range(1, len(data) - 2):
        if data[i] > np.mean([data[i - 1], data[i + 1]]):
            data_despike[i] = np.mean([data[i - 1], data[i + 1]])
        else:
            data_despike[i] = data[i]
    return data_despike


def round_and_despike_df(df: pd.DataFrame) -> pd.DataFrame:
    """Round values and apply per-column despike, then drop rows with NA.

    The function first rounds values to 0 decimal places, then applies the
    module-level ``despike`` function column-wise. Any rows that become NA are
    removed and the index is reset.
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


def subtract_blank_and_clip(df: pd.DataFrame, n_std: float = 1.0, row_nr: int = 31) -> pd.DataFrame:
    """Estimate an instrument blank and subtract it from the DataFrame.

    The blank is estimated per-column from rows ``1:row_nr`` as
    mean + n_std * std. The computed blank is subtracted from every row,
    and the result is clipped to a minimum value of 1 to preserve original
    behaviour.

    Parameters
    ----------
    df:
        Input DataFrame with numeric values.
    n_std:
        Number of standard deviations used when forming the blank.
    row_nr:
        End index (exclusive) of the rows used to estimate the blank.

    Returns
    -------
    pd.DataFrame
        DataFrame with the blank subtracted and clipped values (same columns as input).
    """
    arr = df.values
    means = np.mean(arr[1:row_nr, :], axis=0) + n_std * np.std(arr[1:row_nr, :], axis=0)
    res = (arr - means).clip(1)
    normalized_blank = pd.DataFrame(data=res, columns=df.columns)
    return normalized_blank


def apply_multiple_despike(df: pd.DataFrame, n_des: int = 1) -> pd.DataFrame:
    """Apply the module-level ``despike`` function multiple times column-wise.

    Parameters
    ----------
    df:
        DataFrame to despike.
    n_des:
        Number of despike passes to run.

    Returns
    -------
    pd.DataFrame
        Despiked DataFrame; any rows containing NA are dropped.
    """
    res = df.values
    for _ in range(n_des):
        res = np.apply_along_axis(despike, 0, res)
    despike_df = pd.DataFrame(data=res, columns=df.columns)
    despike_df.dropna(inplace=True)
    return despike_df


def sort_and_bunch(df: pd.DataFrame, n: int = 9, bunch_no: int = 10) -> pd.DataFrame:
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
    """Add a 'name' column derived from the source filename and the row index.

    The filename is extracted by taking the substring after the last '/'
    character and removing the last three characters (mirrors the original
    behaviour which removed a three-character extension).

    Parameters
    ----------
    bunched:
        DataFrame of bunched measurements.
    path:
        Source file path used to extract the filename.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added 'name' column.
    """
    filename = path[path.rfind("/") + 1 : -3]
    bunched["name"] = [f"{filename}_{i}" for i in range(len(bunched))]
    return bunched

def join_inks_and_inds_rows(
    res_all, 
    indicators_suffix = '.i', 
    inks_suffix = '.a', 
    indicators_output_suffix = '_i', 
    inks_output_suffix = '_a',
    sep='_'
) -> pd.DataFrame:

    def remove_ending_number(name, sep=sep):
        res = name.split(sep)
        res = sep.join(res[:-1])
        return res

    res_inds = res_all[res_all['name'].apply(lambda x: remove_ending_number(x).endswith(indicators_suffix))].copy()
    res_inds.columns = res_inds.columns + indicators_output_suffix
    res_inds['name short'] = res_inds['name' + indicators_output_suffix].apply(lambda x: x.replace(indicators_suffix, ''))

    res_inks = res_all[res_all['name'].apply(lambda x: remove_ending_number(x).endswith(inks_suffix))].copy()
    res_inks.columns = res_inks.columns + inks_output_suffix
    res_inks['name short'] = res_inks['name' + inks_output_suffix].apply(lambda x: x.replace(inks_suffix, ''))

    res = pd.merge(res_inks, res_inds, on='name short', how='inner')
    res.drop(columns=['name short'], inplace=True)
    return res

def visualise_intemediate_steps(intermediate_dfs, nrows = 2, figsize=(12,6)):

    dfs = intermediate_dfs
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(intermediate_dfs)//2,
        figsize=figsize,
        sharey='row'
    )

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot each data vector
    for i, ax in enumerate(axes):
        if i == 5:
            ax.plot(dfs[i][el], '*')
        else:
            ax.plot(dfs[i][el])
        ax.set_title(preprocessing_steps[i])
        ax.set_xticks([])

    # Label shared axes
    fig.supxlabel('Time')
    fig.supylabel('Signal ' + el)

    # Improve layout
    plt.tight_layout()
    plt.show()


def machine(
    path: str, 
    elements_dict: dict, 
    n_std: int = 1, 
    row_nr: int = 31, 
    n_des: int = 1, 
    n: int = 10, 
    bunch_no: int = 10, 
    to_fe: bool = False, 
    keep_fe: bool = True,
    intermediate_steps = False
) -> pd.DataFrame:
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
    path :
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
    if intermediate_steps:
        intermediate_dfs = []

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

    if intermediate_steps:

        selected.columns = despike_df.columns
        cleaned.columns = despike_df.columns
        dfs = [selected, cleaned, normalized, normalized_blank, despike_df, bunched]
        return bunched, intermediate_steps

    return bunched

def preprocess_all_from_directory(
    raw_data_path: str,
    elements_dict: dict,
    preprocessed_data_path: str = "",
    n_std: int = 1,
    row_nr: int = 31,
    n_des: int = 1,
    n: int = 10,
    bunch_no: int = 10,
    to_fe: bool = False,
    keep_fe: bool = True,
    inks_present = False,
    indicators_suffix = '.i',
    inks_suffix = '.a',
    indicators_output_suffix = '_i',
    inks_output_suffix = '_a',
    sort=True
) -> pd.DataFrame:
    """Preprocess all raw files in a directory using the ``machine`` pipeline.

    Iterates over files in ``raw_data_path``, runs the ``machine`` pipeline for
    each file and concatenates the resulting bunched DataFrames. Optionally
    writes the combined preprocessed CSV to ``preprocessed_data_path``.

    Parameters
    ----------
    raw_data_path:
        Path to a directory containing raw CSV files. Files are iterated using
        ``os.listdir`` and each name is passed to ``machine`` (the path is
        joined by simple concatenation, so ensure ``raw_data_path`` ends with '/').
    elements_dict:
        Mapping of expected column names to isotope values for normalization.
    preprocessed_data_path:
        If provided, the combined DataFrame will be written to this CSV path.
    n_std, row_nr, n_des, n, bunch_no, to_fe, keep_fe:
        Parameters forwarded to ``machine`` unchanged.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of preprocessed results from all files.
    """

    res_dict = {}

    for raw_data_file in os.listdir(raw_data_path):

        filename = '.'.join(raw_data_file.split('.')[:-1])

        res = machine(
                     path = raw_data_path + raw_data_file,
                     elements_dict = elements_dict, 
                     n_std = n_std, 
                     row_nr = row_nr, 
                     n_des = n_des, 
                     n = n, 
                     bunch_no = bunch_no, 
                     to_fe = to_fe, 
                     keep_fe = keep_fe
                     )
        
        res_dict[filename] = res

    res_all = pd.concat(res_dict.values())
    res_all.reset_index(inplace=True, drop=True)

    if inks_present:
        res_all = join_inks_and_inds_rows(res_all, 
                                        indicators_suffix = indicators_suffix, 
                                        inks_suffix = inks_suffix, 
                                        indicators_output_suffix = indicators_output_suffix, 
                                        inks_output_suffix = inks_output_suffix)

        res_all.dropna(axis=1, inplace=True)

    if sort:
        res_all.sort_values(by='name'+indicators_output_suffix if inks_present else 'name', inplace=True)

    if preprocessed_data_path:

        res_all.to_csv(preprocessed_data_path, index=False)

    return res_all
