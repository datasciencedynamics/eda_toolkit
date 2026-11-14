import os
import gc
import sys
from numpy.testing import verbose
import pandas as pd
import numpy as np
import warnings
from itertools import combinations
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from tqdm import tqdm
from pathlib import Path

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Union

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

from typing import Optional, List, Union, Tuple, Dict, Any

PathLike = Union[str, Path]


if sys.version_info >= (3, 7):
    from datetime import datetime
else:
    import datetime

from ._data_manager_utils import _df_to_markdown, _flag_iqr, _flag_zscore

################################################################################
############################# Path Directories #################################
################################################################################

def ensure_directory(path: PathLike) -> None:
    """
    Ensure that the directory exists. If not, create it.
    """

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory exists: {path}")


################################################################################
######################### Read CSV with Progress Bar ###########################
################################################################################

def read_csv_with_progress(
    file_path,
    nrows=None,
    chunksize=10000,
    low_memory=False,
    **kwargs,
):
    """
    Read a CSV file with a progress bar. Optionally limit to the first
    `nrows` rows.

    Parameters:
    - file_path (str): Path to the CSV file.
    - nrows (int or None): If specified, limits the number of rows read.
    - chunksize (int): Number of rows per chunk. Default is 10,000.
    - low_memory (bool): If False, disables mixed-type inference for
      better dtype consistency at the cost of higher memory usage.
      Default is False.
    - **kwargs: Additional keyword arguments passed to pd.read_csv
      (e.g. sep, encoding, usecols, dtype, skiprows). Note that
      `chunksize` cannot be passed via kwargs — use the explicit
      parameter instead.

    Returns:
    - pd.DataFrame: DataFrame containing the read data.

    Raises:
    - ValueError: If `chunksize` is passed via kwargs.
    """
    if "chunksize" in kwargs:
        raise ValueError(
            "Pass `chunksize` as a direct argument, not via kwargs."
        )

    # Count total lines (minus header)
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f) - 1  # exclude header

    # Suppress DtypeWarning
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    chunks = []
    total_read = 0

    with tqdm(
        total=nrows if nrows else total_lines,
        desc=f"Reading {file_path}",
        unit="line",
    ) as pbar:
        for chunk in pd.read_csv(
            file_path,
            chunksize=chunksize,
            low_memory=low_memory,
            **kwargs,
        ):
            if nrows:
                remaining = nrows - total_read
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)
            chunks.append(chunk)
            read_now = len(chunk)
            total_read += read_now
            pbar.update(read_now)
            if nrows and total_read >= nrows:
                break

    return pd.concat(chunks, ignore_index=True)

################################################################################
############################ Generate Random IDs ###############################
################################################################################


def add_ids(
    df: pd.DataFrame,
    id_colname: str = "ID",
    num_digits: int = 9,
    seed: Optional[int] = None,
    set_as_index: bool = False,
) -> pd.DataFrame:
    """
    Add a column of unique IDs with a specified number of digits to the DataFrame.

    This function ensures all generated IDs are unique, even for large datasets,
    by tracking and resolving potential collisions during ID generation.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to which IDs will be added.
    id_colname : str, optional
        The name of the new column for the unique IDs (default is "ID").
    num_digits : int, optional
        The number of digits for the unique IDs (default is 9). The first digit
        will always be non-zero to ensure proper formatting.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility
        (default is None).
    set_as_index : bool, optional
        If True, the generated ID column will be set as the index of the
        DataFrame (default is False).

    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with a new column of unique IDs. If `set_as_index`
        is True, the new ID column will replace the existing index.

    Raises:
    -------
    ValueError
        If the number of rows in the DataFrame exceeds the pool of possible
        unique IDs for the specified `num_digits`.

    Notes:
    ------
    - The function ensures all IDs are unique by resolving potential collisions
      during generation, even for large datasets.
    - The total pool size of unique IDs is determined by `9 * (10^(num_digits - 1))`,
      since the first digit must be non-zero.
    - If `set_as_index` is False, the ID column will be added as the first column
      in the DataFrame.
    - Warnings are printed if the number of rows in the DataFrame approaches the
      pool size of possible unique IDs, recommending increasing `num_digits`.
    - Setting a random seed ensures reproducibility of the generated IDs.
    """

    # Check if the DataFrame index is unique
    if df.index.is_unique:
        print("The DataFrame index is unique.")
    else:
        print("Warning: The DataFrame index is not unique.")
        print("Duplicate index entries:", df.index[df.index.duplicated()].tolist())

    # Calculate the total pool size of possible unique IDs
    pool_size = 9 * (10 ** (num_digits - 1))  # First digit is non-zero
    n_rows = len(df)

    # Check if the number of rows exceeds or approaches the pool size
    if n_rows > pool_size:
        raise ValueError(
            f"The number of rows ({n_rows}) exceeds the total pool of possible "
            f"unique IDs ({pool_size}). Increase the number of digits to avoid "
            f"this issue."
        )
    elif n_rows > pool_size * 0.9:
        print(
            f"Warning: The number of rows ({n_rows}) is approaching the pool of "
            f"possible unique IDs ({pool_size}). Consider increasing the number "
            f"of digits."
        )

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Initialize a set to track generated IDs for uniqueness
    unique_ids = set()
    ids = []

    while len(ids) < n_rows:
        # Generate random IDs
        first_digits = np.random.choice(list("123456789"), size=n_rows - len(ids))
        other_digits = np.random.choice(
            list("0123456789"), size=(n_rows - len(ids), num_digits - 1)
        )
        batch_ids = [fd + "".join(od) for fd, od in zip(first_digits, other_digits)]

        # Filter out duplicates and add unique IDs
        for id_ in batch_ids:
            if id_ not in unique_ids:
                ids.append(id_)
                unique_ids.add(id_)

    # Assign the unique IDs to the DataFrame
    df[id_colname] = ids

    if set_as_index:
        # Optionally set the new ID column as the index
        df = df.set_index(id_colname)
    else:
        # Ensure the new ID column is the first column
        columns = [id_colname] + [col for col in df.columns if col != id_colname]
        df = df[columns]

    return df


################################################################################
################################# Trailing Periods #############################
################################################################################


def strip_trailing_period(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Remove trailing periods from values in a specified column of a DataFrame.

    This function processes values in the specified column to remove trailing
    periods, handling both strings and numeric values (including those represented
    as strings). The function preserves the original data type wherever possible.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the column to be processed.

    column_name : str
        The name of the column containing values with potential trailing periods.

    Returns:
    --------
    The updated DataFrame with trailing periods removed from the specified column.

    Notes:
    ------
    - For string values, trailing periods are stripped directly.
    - For numeric values represented as strings (e.g., "1234."), the trailing
      period is removed, and the value is converted back to a numeric type if
      possible.
    - NaN values are preserved and remain unprocessed.
    - Non-string and non-numeric types are returned as-is.

    Raises:
    -------
    ValueError
        If the specified `column_name` does not exist in the DataFrame, pandas
        will raise a `ValueError`.
    """

    def fix_value(value):
        # Process only if the value is not NaN
        if pd.notnull(value):
            # Handle strings
            if isinstance(value, str) and value.endswith("."):
                return value.rstrip(".")
            # Handle floats represented as strings
            value_str = str(value)
            if value_str.endswith("."):
                value_str = value_str.rstrip(".")
                try:
                    return float(value_str)  # Convert back to float if possible
                except ValueError:
                    return value_str  # Fallback to string if conversion fails
        return value  # Return as is for NaN or other types

    # Apply the fix_value function to the specified column
    df[column_name] = df[column_name].apply(fix_value)

    return df


################################################################################
########################### Standardized Dates #################################
################################################################################


# Function to parse and standardize date strings based on the new rule
def parse_date_with_rule(date_str: str) -> str:
    """
    Parse and standardize date strings based on the provided rule.

    This function takes a date string and standardizes it to the ISO 8601 format
    (YYYY-MM-DD). It assumes dates are provided in either day/month/year or
    month/day/year format. The function first checks if the first part of the
    date string (day or month) is greater than 12, which unambiguously indicates
    a day/month/year format. If the first part is 12 or less, the function
    attempts to parse the date as month/day/year, falling back to day/month/year
    if the former raises a ValueError due to an impossible date (e.g., month
    being greater than 12).

    Parameters:
        date_str (str): A date string to be standardized.

    Returns:
        str: A standardized date string in the format YYYY-MM-DD.

    Raises:
        ValueError: If date_str is in an unrecognized format or if the function
        cannot parse the date.
    """

    parts = date_str.split("/")
    # If the first part is greater than 12, it can only be a day, thus d/m/Y
    if int(parts[0]) > 12:
        return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
    # Otherwise, try both formats where ambiguity exists
    else:
        try:
            return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            return datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")


################################################################################
############################### DataFrame Profiler #############################
################################################################################


def dataframe_profiler(
    df: pd.DataFrame,
    background_color: Optional[str] = None,
    return_df: bool = False,
    sort_cols_alpha: bool = False,
) -> Union[pd.DataFrame, "Styler"]:
    """
    Analyze DataFrame columns to provide a profile of summary statistics such as
    data type, null counts, unique values, and most frequent values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze.

    background_color : str, optional
        Hex color code or color name for background styling in the output
        DataFrame. Applies to specific columns such as unique value totals
        and percentages. Defaults to None.

    return_df : bool, optional
        If True, returns the plain DataFrame with the summary statistics.
        If False, returns a styled DataFrame for visual presentation
        (default). In terminal environments, always returns the plain
        DataFrame regardless of this setting.

    sort_cols_alpha : bool, optional
        If True, sorts columns in alphabetical order before returning the
        DataFrame. Applies to both plain and styled outputs. Defaults to False.

    Returns:
    --------
    pandas.DataFrame
        - If `return_df` is True or the function is running in a terminal
          environment, returns the plain DataFrame containing column summary
          statistics.
        - Otherwise, returns a styled DataFrame with optional background color
          for specific columns, when running in a Jupyter Notebook.

        The summary DataFrame includes the following columns:
        - column: Column name
        - dtype: Data type of the column
        - null_total: Total number of null values
        - null_pct: Percentage of null values
        - unique_values_total: Total number of unique values
        - max_unique_value: The most frequent value
        - max_unique_value_total: Frequency of the most frequent value
        - max_unique_value_pct: Percentage of the most frequent value

    Notes:
    ------
    - The function automatically detects whether it is running in a Jupyter
      Notebook (using `ipykernel`) or a terminal environment and adjusts the
      output accordingly.
    - In Jupyter environments, it attempts to style the output using Pandas'
      Styler. If `hide` is deprecated in the installed Pandas version, it uses
      `hide_index` as a fallback.
    - The function uses a `tqdm` progress bar to indicate the processing
      status of columns.
    - NaN values and empty strings are preprocessed to ensure consistent
      handling in the summary statistics.
    """

    print("Shape: ", df.shape, "\n")
    start_time = (
        datetime.now() if sys.version_info >= (3, 7) else datetime.datetime.now()
    )

    # Convert dbdate dtype to datetime
    for col in df.columns:
        if df[col].dtype == "dbdate":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Null pre-processing with Pandas NA
    df = df.fillna(pd.NA)
    # Replace empty strings with Pandas NA
    df = df.apply(
        lambda col: col.map(
            lambda x: pd.NA if isinstance(x, str) and x == "" else x,
        )
    )
    # Begin Process...
    columns_value_counts = []

    # Wrap the column iteration in tqdm for a progress bar
    for col in tqdm(df.columns, desc="Processing columns"):
        col_str = df[col].astype(str).replace("<NA>", "null").replace("NaT", "null")
        value_counts = col_str.value_counts()
        max_unique_value = value_counts.index[0]
        max_unique_value_total = value_counts.iloc[0]
        columns_value_counts.append(
            {
                "column": col,
                "dtype": df[col].dtype,
                "null_total": df[col].isnull().sum(),
                "null_pct": round(df[col].isnull().sum() / df.shape[0] * 100, 2),
                "unique_values_total": df[col].nunique(),
                "max_unique_value": max_unique_value,
                "max_unique_value_total": max_unique_value_total,
                "max_unique_value_pct": round(
                    max_unique_value_total / df.shape[0] * 100, 2
                ),
            }
        )
    stop_time = (
        datetime.now() if sys.version_info >= (3, 7) else datetime.datetime.now()
    )
    print()
    print(
        "Total seconds of processing time:",
        (stop_time - start_time).total_seconds(),
    )
    print()
    if sort_cols_alpha:
        result_df = pd.DataFrame(columns_value_counts).sort_values(by="column")
    else:
        result_df = pd.DataFrame(columns_value_counts)

    # Detect environment (Jupyter Notebook or terminal)
    is_notebook_env = "ipykernel" in sys.modules

    if return_df or not is_notebook_env:
        # Return the plain DataFrame for terminal environments or if explicitly requested
        return result_df
    else:
        ## Output, try/except, accounting for the potential of Python version with
        ## the styler as hide_index() is deprecated since Pandas 1.4, in such cases,
        ## hide() is used instead
        # Return the styled DataFrame for Jupyter environments
        if sort_cols_alpha:
            # Sort the DataFrame alphabetically before styling
            try:
                styled_result = (
                    result_df.sort_values(by="column")
                    .style.hide()
                    .format(precision=2)
                    .set_properties(
                        subset=[
                            "unique_values_total",
                            "max_unique_value",
                            "max_unique_value_total",
                            "max_unique_value_pct",
                        ],
                        **(
                            {"background-color": background_color}
                            if background_color
                            else {}
                        ),
                    )
                )
            except AttributeError:
                # Fallback for Pandas versions where `hide()` is deprecated
                styled_result = (
                    result_df.sort_values(by="column")
                    .style.hide_index()
                    .format(precision=2)
                    .set_properties(
                        subset=[
                            "unique_values_total",
                            "max_unique_value",
                            "max_unique_value_total",
                            "max_unique_value_pct",
                        ],
                        **(
                            {"background-color": background_color}
                            if background_color
                            else {}
                        ),
                    )
                )
        else:
            # Do not sort columns alphabetically
            try:
                styled_result = (
                    result_df.style.hide()
                    .format(precision=2)
                    .set_properties(
                        subset=[
                            "unique_values_total",
                            "max_unique_value",
                            "max_unique_value_total",
                            "max_unique_value_pct",
                        ],
                        **(
                            {"background-color": background_color}
                            if background_color
                            else {}
                        ),
                    )
                )
            except AttributeError:
                # Fallback for Pandas versions where `hide()` is deprecated
                styled_result = (
                    result_df.style.hide_index()
                    .format(precision=2)
                    .set_properties(
                        subset=[
                            "unique_values_total",
                            "max_unique_value",
                            "max_unique_value_total",
                            "max_unique_value_pct",
                        ],
                        **(
                            {"background-color": background_color}
                            if background_color
                            else {}
                        ),
                    )
                )

        return styled_result


################################################################################
############################ Summarize All Combinations ########################
################################################################################


def summarize_all_combinations(
    df: pd.DataFrame,
    variables: List[str],
    data_path: Union[str, Path] = None,
    data_name: str = None,
    min_length: int = 2,
) -> Tuple[Dict[Tuple[str, ...], pd.DataFrame], List[Tuple[str, ...]]]:
    """
    Generate summary tables for all possible combinations of the specified
    variables in the DataFrame and optionally save them to an Excel file
    with detailed formatting.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    variables : list of str
        List of column names from the DataFrame to generate combinations.

    data_path : str, optional
        Directory path where the output Excel file will be saved. Required
        only if saving the results to a file.

    data_name : str, optional
        Name of the output Excel file. Required only if saving the results
        to a file.

    min_length : int, optional (default=2)
        Minimum size of the combinations to generate.

    Returns:
    --------
    summary_tables : dict
        A dictionary where keys are tuples of column names (combinations) and
        values are the corresponding summary DataFrames.

    all_combinations : list of tuple
        A list of all generated combinations, where each combination is
        represented as a tuple of column names.

    Notes:
    ------
    - **Combination Generation**:
        - Generates all combinations of the specified variables with a size
          greater than or equal to `min_length`.
        - Uses **`tqdm`** for progress tracking during combination generation.
    - **Excel Output (optional)**:
        - If both `data_path` and `data_name` are provided, results are saved
          to an Excel file with a "Table of Contents" sheet linking to each
          combination sheet.
        - Sheet names are truncated to 31 characters to meet Excel's limitations.
    - **Formatting**:
        - Headers in all sheets are bold, left-aligned, and borderless.
        - Columns are auto-fitted based on content length for improved readability.
        - A left-aligned format is applied to all columns.
    - **Progress Tracking**:
        - The function uses **`tqdm`** progress bars for tracking combination
          generation, writing the Table of Contents, and writing summary tables
          to Excel.

    Raises:
    -------
    ValueError
        If the `variables` list is empty or not provided.

    Outputs:
    --------
    - If `data_path` and `data_name` are provided:
        - An Excel file at the specified path with the following:
            - A "Table of Contents" sheet linking to all combination sheets.
            - Individual sheets for each variable combination summarizing counts
              and proportions.
    - If not provided:
        - The function still returns all generated summary tables and combinations
          without writing any files.
    """

    summary_tables = {}
    grand_total = len(df)
    all_combinations = []

    df_copy = df.copy()

    # Calculate total number of combinations for smoother tqdm updates
    total_combinations = sum(
        len(list(combinations(variables, i)))
        for i in range(min_length, len(variables) + 1)
    )

    # First tqdm for combination generation
    with tqdm(total=total_combinations, desc="Generating combinations") as pbar:
        for i in range(min_length, len(variables) + 1):
            for combination in combinations(variables, i):
                all_combinations.append(combination)
                for col in combination:
                    df_copy[col] = df_copy[col].astype(str)

                count_df = (
                    df_copy.groupby(list(combination)).size().reset_index(name="Count")
                )
                count_df["Proportion"] = (count_df["Count"] / grand_total * 100).fillna(
                    0
                )
                summary_tables[tuple(combination)] = count_df

                # Update progress bar manually for smoother updates
                pbar.update(1)

    sheet_names = [
        ("_".join(combination)[:31]) for combination in summary_tables.keys()
    ]
    descriptions = [
        "Summary for " + ", ".join(combination) for combination in summary_tables.keys()
    ]
    legend_df = pd.DataFrame(
        {"Sheet Name": sheet_names, "Description": descriptions},
    )

    file_path = f"{data_path}/{data_name}"

    if data_path and data_name:
        file_path = Path(data_path) / data_name
        # Writing to Excel with progress tracking
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            # Write the Table of Contents (legend sheet)
            legend_df.to_excel(writer, sheet_name="Table of Contents", index=False)

            workbook = writer.book
            toc_worksheet = writer.sheets["Table of Contents"]

            # Add hyperlinks to the sheet names
            for i, sheet_name in enumerate(
                tqdm(sheet_names, desc="Writing Table of Contents", leave=False),
                start=2,
            ):
                cell = f"A{i}"
                toc_worksheet.write_url(
                    cell, f"internal:'{sheet_name}'!A1", string=sheet_name
                )

            # Set column widths and alignment for Table of Contents
            toc_worksheet.set_column("A:A", 50)  # Set width for column A (Sheet Name)
            toc_worksheet.set_column("B:B", 100)  # Set width for column B (Description)

            # Create a format for left-aligned text
            cell_format = workbook.add_format({"align": "left"})
            toc_worksheet.set_column("A:A", 50, cell_format)  # Column A
            toc_worksheet.set_column("B:B", 100, cell_format)  # Column B

            # Format the header row of Table of Contents
            header_format_toc = workbook.add_format(
                {"bold": True, "align": "left", "border": 0}
            )
            toc_worksheet.write_row("A1", legend_df.columns, header_format_toc)

            # Define a format with no borders for the header row in other sheets
            header_format_no_border = workbook.add_format(
                {"bold": True, "border": 0, "align": "left"}
            )

            # Define a format for left-aligned text in other sheets
            left_align_format = workbook.add_format({"align": "left"})

            # Third tqdm for writing summary tables
            for sheet_name, table in tqdm(
                summary_tables.items(),
                desc="Writing summary tables",
                leave=True,
            ):
                sheet_name_str = "_".join(sheet_name)[
                    :31
                ]  # Ensure sheet name is <= 31 characters
                table.to_excel(writer, sheet_name=sheet_name_str, index=False)

                worksheet = writer.sheets[sheet_name_str]

                # Apply format to the header row (top row)
                for col_num, col_name in enumerate(table.columns):
                    worksheet.write(0, col_num, col_name, header_format_no_border)

                # Apply left alignment to all columns
                for row_num in range(1, len(table) + 1):
                    for col_num in range(len(table.columns)):
                        worksheet.write(
                            row_num,
                            col_num,
                            table.iloc[row_num - 1, col_num],
                            left_align_format,
                        )

                # Auto-fit all columns with added space
                for col_num, col_name in enumerate(table.columns):
                    max_length = max(
                        table[col_name].astype(str).map(len).max(), len(col_name)
                    )
                    worksheet.set_column(
                        col_num, col_num, max_length + 2, left_align_format
                    )
        # Add the Writing to Excel progress bar after everything else
        with tqdm(desc="Finalizing Excel file", total=1, leave=True) as pbar:
            pbar.update(1)

        print(f"Data saved to {file_path}")

    else:
        print("Excel export skipped (no path or filename provided).")

    return summary_tables, all_combinations


################################################################################
############################ Save DataFrames to Excel ##########################
################################################################################


def save_dataframes_to_excel(
    file_path: Union[str, Path],
    df_dict: Dict[str, pd.DataFrame],
    decimal_places: int = 0,
) -> None:
    """
    Save multiple DataFrames to separate sheets in an Excel file with customized
    formatting, including column autofit, numeric formatting, and progress
    tracking.

    Parameters:
    -----------
    file_path : str
        Full path to the output Excel file.

    df_dict : dict
        Dictionary where keys are sheet names and values are DataFrames to save
        to those sheets.

    decimal_places : int, optional (default=0)
        Number of decimal places to round numeric columns. If set to 0, numeric
        columns will be saved as integers.

    Notes:
    ------
    - Columns are automatically adjusted to fit their content and left-aligned.
    - Numeric columns are rounded to the specified decimal places and formatted
      accordingly. If `decimal_places` is 0, numeric columns are saved as integers.
    - Headers are bold, left-aligned, and have no borders.
    - The function uses **`tqdm`** to display a progress bar for tracking the
      saving of DataFrames to Excel sheets.
    - This function requires the `xlsxwriter` library for writing Excel files.
    - Non-numeric columns are left-aligned by default.

    Raises:
    -------
    ValueError
        If `df_dict` is empty or not provided, a `ValueError` will be raised.
    """

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Customize header format (remove borders)
        header_format = workbook.add_format(
            {
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "border": 0,  # Remove borders
                "align": "left",  # Left align
            }
        )

        # Customize cell format (left align)
        cell_format_left = workbook.add_format({"align": "left"})  # Left align

        # Customize number format based on decimal_places
        if decimal_places == 0:
            number_format_str = "0"
            cell_format_number = workbook.add_format(
                {
                    "align": "left",
                    "num_format": number_format_str,
                }  # Left align  # Number format
            )
        else:
            number_format_str = f"0.{decimal_places * '0'}"
            cell_format_number = workbook.add_format(
                {
                    "align": "left",
                    "num_format": number_format_str,
                }  # Left align  # Number format
            )

        # Write each DataFrame to its respective sheet
        with tqdm(total=len(df_dict), desc="Saving DataFrames", leave=True) as pbar:
            for sheet_name, df in df_dict.items():
                # Round numeric columns to the specified number of decimal places
                df = df.round(decimal_places)
                if decimal_places == 0:
                    df = df.apply(
                        lambda x: (
                            x.astype(int) if pd.api.types.is_numeric_dtype(x) else x
                        )
                    )
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]

                # Write header with custom format
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Auto-fit all columns with added space
                for col_num, col_name in enumerate(df.columns):
                    max_length = max(
                        df[col_name].astype(str).map(len).max(),
                        len(col_name),
                    )
                    # Determine if the column is numeric by dtype
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        worksheet.set_column(
                            col_num, col_num, max_length + 2, cell_format_number
                        )
                    else:
                        worksheet.set_column(
                            col_num, col_num, max_length + 2, cell_format_left
                        )

                pbar.update(1)  # Update the progress bar for each sheet

    print(f"DataFrames saved to {file_path}")


################################################################################
############################## Table 1 Generator ###############################
################################################################################


def table1_to_str(
    df: pd.DataFrame,
    float_precision: int = 2,
    max_col_width: int = 18,
    padding: int = 1,
) -> str:
    """
    Pretty-print a summary table (like Table 1) with clean alignment.
    """
    if df is None or df.empty:
        return "[Empty Table]"

    def format_val(val):
        if pd.isna(val):
            return ""
        elif isinstance(val, float):
            return f"{val:.{float_precision}f}"
        else:
            return str(val)

    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(format_val)

    col_widths = {
        col: min(max(len(col), formatted[col].str.len().max()), max_col_width)
        for col in formatted.columns
    }

    def pad(val, width):
        val = val[:width]
        return f"{' ' * padding}{val:<{width}}{' ' * padding}"

    header = "|".join([pad(col, col_widths[col]) for col in formatted.columns])
    separator = "|".join(
        ["-" * (col_widths[col] + 2 * padding) for col in formatted.columns]
    )
    rows = [
        "|".join([pad(val, col_widths[col]) for col, val in row.items()])
        for _, row in formatted.iterrows()
    ]

    return "\n".join([header, separator] + rows)


class TableWrapper:
    """
    Wraps a DataFrame to override its string output without affecting
    Jupyter display. Preserves pretty-print when modifying the DataFrame.
    """

    def __init__(self, df: pd.DataFrame, string: str) -> None:
        self._df = df
        self._string = string

    def __str__(self) -> str:
        return self._string

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._df, attr)

    def __getitem__(self, key: Any) -> Any:
        return self._df[key]

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        return iter(self._df)

    def drop(self, *args: Any, **kwargs: Any) -> "TableWrapper":
        """
        Allows dropping rows or columns while preserving pretty-print formatting.
        """
        dropped_df = self._df.drop(*args, **kwargs)
        return TableWrapper(dropped_df, table1_to_str(dropped_df))


def generate_table1(
    df: pd.DataFrame,
    apply_bonferroni: bool = False,
    apply_bh_fdr: bool = False,
    categorical_cols: Optional[List[str]] = None,
    continuous_cols: Optional[List[str]] = None,
    decimal_places: int = 2,
    export_markdown: bool = False,
    drop_columns: Optional[Union[str, List[str]]] = None,
    drop_variables: Optional[Union[str, List[str]]] = None,
    markdown_path: Optional[Union[str, Path]] = None,
    max_categories: Optional[int] = None,
    detect_binary_numeric: bool = True,
    return_markdown_only: bool = False,
    value_counts: bool = False,
    include_types: str = "both",  # "continuous", "categorical", or "both"
    combine: bool = True,
    groupby_col: Optional[str] = None,
    use_fisher_exact: bool = False,
    use_welch: bool = True,
    proportion_relative_to_cohort: bool = True,
    include_overall: bool = True,
) -> Union[
    pd.DataFrame,
    Tuple[pd.DataFrame, pd.DataFrame],
    str,
    Dict[str, str],
    Tuple[str, str],
]:
    """
    Generate a summary table (Table 1) for a given DataFrame.
    """

    if apply_bonferroni and apply_bh_fdr:
        raise ValueError(
            f"Cannot apply both Bonferroni and Benjamini-Hochberg corrections "
            f"simultaneously. Choose one."
        )

    valid_types = {"categorical", "continuous", "both"}
    if include_types not in valid_types:
        raise ValueError(
            f"Invalid include_types: '{include_types}'. Must be one of {valid_types}."
        )

    if isinstance(drop_columns, str):
        drop_columns = [drop_columns]
    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    if continuous_cols is None:
        continuous_cols = (
            df.select_dtypes(include=["number"])
            .columns.difference(categorical_cols)
            .tolist()
        )

    # slice copy [:] protects against skipping elements during in-place
    # modification of continuous_cols while iterating
    if detect_binary_numeric:
        for col in continuous_cols[:]:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2:
                categorical_cols.append(col)
                continuous_cols.remove(col)

    total_rows = len(df)
    continuous_parts = []
    categorical_parts = []

    if groupby_col:
        group_vals = df[groupby_col].dropna().unique()
        if len(group_vals) != 2:
            raise ValueError(
                "groupby_col must have exactly two groups for p-value calculation."
            )
        g1, g2 = group_vals
        group1_label = f"{g1} (n = {len(df[df[groupby_col] == g1]):,})"
        group2_label = f"{g2} (n = {len(df[df[groupby_col] == g2]):,})"

    if include_types in ["continuous", "both"]:
        for col in continuous_cols:
            series = df[col]
            non_missing = series.dropna()
            row = {
                "Variable": col,
                "Type": "Continuous",
                "Mean": round(non_missing.mean(), decimal_places),
                "SD": round(non_missing.std(), decimal_places),
                "Median": round(non_missing.median(), decimal_places),
                "Min": round(non_missing.min(), decimal_places),
                "Max": round(non_missing.max(), decimal_places),
                "Mode": (
                    round(non_missing.mode().iloc[0], decimal_places)
                    if not non_missing.mode().empty
                    else ""
                ),
                "Missing (n)": series.isna().sum(),
                "Missing (%)": 100 * series.isna().mean(),
                "Count": non_missing.count(),
                "Proportion (%)": 100 * non_missing.count() / total_rows,  
            }
            if groupby_col:
                x1 = df[df[groupby_col] == g1][col].dropna()
                x2 = df[df[groupby_col] == g2][col].dropna()
                if use_welch:
                    print(f"Using Welch's t-test for continuous variable: {col}")
                    _, p = ttest_ind(x1, x2, equal_var=False)
                else:
                    print(f"Using Student's t-test for continuous variable: {col}")
                    _, p = ttest_ind(x1, x2, equal_var=True)

                row[group1_label] = (
                    f"{x1.mean():.{decimal_places}f} "
                    f"({x1.std():.{decimal_places}f})"
                )
                row[group2_label] = (
                    f"{x2.mean():.{decimal_places}f} "
                    f"({x2.std():.{decimal_places}f})"
                )
                row["_raw_pval"] = p
                row["P-value"] = round(p, decimal_places)
            continuous_parts.append(row)

    if include_types in ["categorical", "both"]:
        for col in categorical_cols:
            series = df[col]
            missing_n = series.isna().sum()
            missing_pct = 100 * missing_n / total_rows
            mode_val = series.mode().iloc[0] if not series.mode().empty else ""

            if groupby_col:
                if series.dropna().empty:
                    summary_row = {
                        "Variable": col, "Type": "Categorical",
                        "Mean": "", "SD": "", "Median": "", "Min": "", "Max": "",
                        "Mode": mode_val,
                        "Missing (n)": missing_n, "Missing (%)": missing_pct,
                        "Count": 0, "Proportion (%)": 0,
                        group1_label: "0 (0.00%)",
                        group2_label: "0 (0.00%)",
                        "P-value": "",
                    }
                    categorical_parts.append(summary_row)
                    continue

                ct = pd.crosstab(df[col], df[groupby_col])
                ## warn instead of silently dropping non-2-column tables
                if ct.shape[1] != 2:
                    import warnings
                    warnings.warn(
                        f"Skipping '{col}': crosstab has {ct.shape[1]} group "
                        f"columns, expected 2. Check `groupby_col` values.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                if use_fisher_exact and ct.shape[0] == 2:
                    print(f"Using Fisher's Exact Test for variable: {col}")
                    _, p = fisher_exact(ct.to_numpy())
                else:
                    if use_fisher_exact:
                        print(
                            f"Fisher's Exact Test requires a 2x2 table. "
                            f"Falling back to chi-squared for '{col}'."
                        )
                    else:
                        print(f"Using Chi-squared test for variable: {col}")
                    _, p, _, _ = chi2_contingency(ct)

                summary_row = {
                    "Variable": col,
                    "Type": "Categorical",
                    "Mean": "",
                    "SD": "",
                    "Median": "",
                    "Min": "",
                    "Max": "",
                    "Mode": mode_val,
                    "Missing (n)": missing_n,
                    "Missing (%)": missing_pct,
                    "Count": series.notna().sum(),
                    "Proportion (%)": 100 * series.notna().sum() / total_rows,
                    group1_label: (
                        f"{(df[groupby_col] == g1).sum():,} "
                        f"({100 * (df[groupby_col] == g1).sum() / total_rows:.{decimal_places}f}%)"
                        if g1 in ct.columns
                        else "0"
                    ),
                    group2_label: (
                        f"{(df[groupby_col] == g2).sum():,} "
                        f"({100 * (df[groupby_col] == g2).sum() / total_rows:.{decimal_places}f}%)"
                        if g2 in ct.columns
                        else "0"
                    ),
                    "_raw_pval": p,
                    "P-value": round(p, decimal_places),
                }
                categorical_parts.append(summary_row)

            else:
                summary_row = {
                    "Variable": col,
                    "Type": "Categorical",
                    "Mean": "",
                    "SD": "",
                    "Median": "",
                    "Min": "",
                    "Max": "",
                    "Mode": mode_val,
                    "Missing (n)": missing_n,
                    "Missing (%)": missing_pct,
                    "Count": series.notna().sum(),
                    "Proportion (%)": 100 * series.notna().sum() / total_rows,
                }
                categorical_parts.append(summary_row)

            if value_counts:
                counts = series.value_counts(dropna=False)
                if max_categories:
                    counts = counts.head(max_categories)
                for cat_val, count in counts.items():
                    label = (
                        f"{col} = {cat_val}"
                        if pd.notna(cat_val)
                        else f"{col} = NaN"
                    )
                    if groupby_col:
                        if pd.isna(cat_val):
                            col_mask = df[col].isna()
                        else:
                            col_mask = df[col] == cat_val
                        g1_mask = col_mask & (df[groupby_col] == g1)
                        g2_mask = col_mask & (df[groupby_col] == g2)
                        g1_count = g1_mask.sum()
                        g2_count = g2_mask.sum()
                        g1_total = (df[groupby_col] == g1).sum()
                        g2_total = (df[groupby_col] == g2).sum()
                        g1_prop = 100 * g1_count / g1_total if g1_total else 0
                        g2_prop = 100 * g2_count / g2_total if g2_total else 0
                        g1_str = f"{g1_count:,} ({g1_prop:.{decimal_places}f}%)"
                        g2_str = f"{g2_count:,} ({g2_prop:.{decimal_places}f}%)"
                        row = {
                            "Variable": label,
                            "Type": "Categorical",
                            "Mean": "",
                            "SD": "",
                            "Median": "",
                            "Min": "",
                            "Max": "",
                            "Mode": mode_val,
                            "Missing (n)": missing_n,
                            "Missing (%)": missing_pct,
                            "Count": count,
                            "Proportion (%)": (
                                100 * count / total_rows
                                if proportion_relative_to_cohort
                                else 100 * count / series.notna().sum()
                            ),
                            group1_label: g1_str,
                            group2_label: g2_str,
                        }
                    else:
                        row = {
                            "Variable": label,
                            "Type": "Categorical",
                            "Mean": "",
                            "SD": "",
                            "Median": "",
                            "Min": "",
                            "Max": "",
                            "Mode": mode_val,
                            "Missing (n)": missing_n,
                            "Missing (%)": missing_pct,
                            "Count": count,
                            "Proportion (%)": (
                                100 * count / total_rows
                                if proportion_relative_to_cohort
                                else 100 * count / series.notna().sum()
                            ),
                        }
                    categorical_parts.append(row)

    if apply_bonferroni or apply_bh_fdr:
        all_pval_keys = []
        all_raw_pvals = []

        for i, row in enumerate(continuous_parts):
            if "_raw_pval" in row:
                all_pval_keys.append(("continuous", i))
                all_raw_pvals.append(row["_raw_pval"])

        for i, row in enumerate(categorical_parts):
            if "_raw_pval" in row:
                all_pval_keys.append(("categorical", i))
                all_raw_pvals.append(row["_raw_pval"])

        if apply_bonferroni:
            corrected = [min(p * len(all_raw_pvals), 1.0) for p in all_raw_pvals]
        elif apply_bh_fdr:
            sorted_indices = np.argsort(all_raw_pvals)
            sorted_pvals = np.array(all_raw_pvals)[sorted_indices]
            n = len(sorted_pvals)
            bh_adjusted = np.empty(n)
            for i in range(n):
                bh_adjusted[i] = sorted_pvals[i] * n / (i + 1)
            bh_adjusted = np.minimum.accumulate(bh_adjusted[::-1])[::-1]
            bh_adjusted = np.clip(bh_adjusted, 0, 1.0)
            corrected = np.empty_like(bh_adjusted)
            corrected[sorted_indices] = bh_adjusted
        else:
            corrected = all_raw_pvals

        for (section, i), p_adj in zip(all_pval_keys, corrected):
            if section == "continuous":
                continuous_parts[i]["P-value"] = p_adj
                del continuous_parts[i]["_raw_pval"]
            else:
                categorical_parts[i]["P-value"] = p_adj
                del categorical_parts[i]["_raw_pval"]

    for row in continuous_parts + categorical_parts:
        row.pop("_raw_pval", None)

    if include_overall:     
        def make_overall(row):
            if row.get("Type") == "Continuous":
                mean = row.get("Mean", "")
                sd = row.get("SD", "")
                if mean != "" and sd != "":
                    return (
                        f"{mean:,.{decimal_places}f} "
                        f"({sd:,.{decimal_places}f})"
                    )
                return ""
            elif row.get("Type") == "Categorical":
                count = row.get("Count", "")
                prop = row.get("Proportion (%)", "")
                if count != "" and prop != "":
                    return f"{int(count):,} ({prop:.{decimal_places}f}%)"
                return ""
            return ""

        overall_continuous = [make_overall(r) for r in continuous_parts]
        overall_categorical = [make_overall(r) for r in categorical_parts]

    
    df_continuous = pd.DataFrame(continuous_parts).replace({np.nan: ""})
    df_categorical = pd.DataFrame(categorical_parts).replace({np.nan: ""})

    def format_numeric_cols(df):
            float_cols = [
                "Mean", "SD", "Median", "Min", "Max", "Mode",
                "Missing (%)", "Proportion (%)", "P-value",
            ]
            int_cols = ["Count", "Missing (n)"]

            def is_numeric_string(x):
                try:
                    return x != "" and float(str(x).replace(",", "")) == float(
                        str(x).replace(",", "")
                    )
                except Exception:
                    return False

            for col in float_cols:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: (
                            f"{float(str(x).replace(',', '')):,.{decimal_places}f}"
                            if is_numeric_string(x)
                            else "" if pd.isna(x) else x
                        )
                    )

            for col in int_cols:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: (
                            f"{int(float(str(x).replace(',', ''))):,}"
                            if is_numeric_string(x)
                            else "" if pd.isna(x) else x
                        )
                    )

            return df
    
    df_continuous = format_numeric_cols(df_continuous)
    df_categorical = format_numeric_cols(df_categorical)

    if include_overall:
        if not df_continuous.empty:
            df_continuous.insert(1, "Overall", overall_continuous)
        if not df_categorical.empty:
            df_categorical.insert(1, "Overall", overall_categorical)

    drop_cols = ["Mean", "SD", "Median", "Min", "Max"]
    df_categorical.drop(columns=drop_cols, inplace=True, errors="ignore")

    if drop_columns:
        df_continuous = df_continuous.drop(columns=drop_columns, errors="ignore")
        df_categorical = df_categorical.drop(columns=drop_columns, errors="ignore")

    if drop_variables:
        if not df_continuous.empty and "Variable" in df_continuous.columns:
            df_continuous = df_continuous[
                ~df_continuous["Variable"]
                .astype(str)
                .apply(
                    lambda x: any(
                        x == var or x.startswith(f"{var} =")
                        for var in drop_variables
                    )
                )
            ]
        if not df_categorical.empty and "Variable" in df_categorical.columns:
            df_categorical = df_categorical[
                ~df_categorical["Variable"]
                .astype(str)
                .apply(
                    lambda x: any(
                        x == var or x.startswith(f"{var} =")
                        for var in drop_variables
                    )
                )
            ]

    if export_markdown:
        if not markdown_path:
            markdown_path = "table1.md"
        else:
            markdown_path = str(markdown_path)

        if include_types == "continuous":
            markdown_str = _df_to_markdown(df_continuous)
            with open(markdown_path.replace(".md", "_continuous.md"), "w") as f:
                f.write(markdown_str)
            if return_markdown_only:
                return markdown_str
        elif include_types == "categorical":
            markdown_str = _df_to_markdown(df_categorical)
            with open(markdown_path.replace(".md", "_categorical.md"), "w") as f:
                f.write(markdown_str)
            if return_markdown_only:
                return markdown_str
        else:
            md_cont = _df_to_markdown(df_continuous)
            md_cat = _df_to_markdown(df_categorical)
            with open(markdown_path.replace(".md", "_continuous.md"), "w") as f:
                f.write(md_cont)
            with open(markdown_path.replace(".md", "_categorical.md"), "w") as f:
                f.write(md_cat)
            if return_markdown_only:
                return {"continuous": md_cont, "categorical": md_cat}

    if include_types == "continuous":
        result = df_continuous
    elif include_types == "categorical":
        result = df_categorical
    else:
        result = (
            pd.concat([df_continuous, df_categorical], ignore_index=True)
            if combine
            else (df_continuous, df_categorical)
        )

    if not combine:
        if isinstance(result, tuple):
            result = tuple(r.fillna("") for r in result)
    else:
        result = result.fillna("")

    if return_markdown_only:
        if include_types == "continuous":
            return _df_to_markdown(df_continuous)
        elif include_types == "categorical":
            return _df_to_markdown(df_categorical)
        else:
            return {
                "continuous": _df_to_markdown(df_continuous),
                "categorical": _df_to_markdown(df_categorical),
            }

    if isinstance(result, pd.DataFrame):
        pretty = table1_to_str(result, float_precision=decimal_places)
        return TableWrapper(result, pretty)

    if isinstance(result, tuple):
        cont, cat = result
        pretty_cont = table1_to_str(cont, float_precision=decimal_places)
        pretty_cat = table1_to_str(cat, float_precision=decimal_places)
        return TableWrapper(cont, pretty_cont), TableWrapper(cat, pretty_cat)

    return result

################################################################################
############################## Contingency Table ###############################
################################################################################


def contingency_table(
    df: pd.DataFrame,
    cols: Optional[Union[str, List[str]]] = None,
    sort_by: int = 0,
) -> pd.DataFrame:
    """
    Create a contingency table from one or more columns in a DataFrame, with
    options to sort the results.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to analyze.

    cols : str or list of str, optional
        The name of a single column (as a string) or a list of column names
        for multiple columns. At least one column must be provided.

    sort_by : int, optional (default=0)
        Sorting option for the results. Enter 0 to sort the results by the
        column group(s) specified in `cols`, or enter 1 to sort the results
        by the 'Total' column in descending order.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the contingency table with the specified columns,
        a 'Total' column representing the count of occurrences, and a
        'Percentage' column representing the percentage of the total count.

    Raises:
    -------
    ValueError
        If no columns are specified in `cols`, or if `sort_by` is not 0 or 1.

    """

    # Ensure at least one column is specified
    if not cols or (isinstance(cols, list) and not cols):
        raise ValueError("At least one DataFrame column must be specified.")

    # Ensure sort_by is either 0 or 1
    if sort_by not in [0, 1]:
        raise ValueError("`sort_by` must be 0 or 1.")

    # Convert single column to list
    if isinstance(cols, str):
        cols = [cols]

    # Ensure df is an independent copy to avoid SettingWithCopyWarning
    df = df.copy(deep=True)

    # Convert categorical columns to string to avoid fillna issue
    for col in cols:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype(str)

    # Convert all values in dataframe to object
    # then fill NA values in the dataframe with empty spaces
    df = df.astype(str).fillna("")

    # Create the contingency table with observed=True
    cont_df = (
        df.groupby(cols, observed=True)
        .size()
        .reset_index(
            name="Total",
        )
    )

    # Calculate the percentage
    cont_df["Percentage"] = 100 * cont_df["Total"] / len(df)

    # Sort values based on provided sort_by parameter
    if sort_by == 0:
        cont_df = cont_df.sort_values(by=cols)
    elif sort_by == 1:
        cont_df = cont_df.sort_values(by="Total", ascending=False)

    # Convert categorical columns to string to avoid fillna issue
    cont_df[cols] = cont_df[cols].astype(str)

    # Results for all groups
    all_groups = pd.DataFrame(
        [
            {
                **{col: "" for col in cols},
                "Total": cont_df["Total"].sum(),
                "Percentage": cont_df["Percentage"].sum(),
            }
        ]
    )

    # Combine results
    c_table = pd.concat(
        [cont_df.fillna(""), all_groups.fillna("")],
        ignore_index=True,
    )

    # Update GroupPct to reflect as a percentage rounded to 2 decimal places
    c_table["Percentage"] = c_table["Percentage"].round(2)

    return c_table


################################################################################
############################## Highlight DF Tables #############################
################################################################################


def highlight_columns(
    df: pd.DataFrame,
    columns: List[str],
    color: str = "yellow",
) -> "pd.io.formats.style.Styler":
    """
    Highlight specific columns in a DataFrame with a specified background color.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be styled.
    columns : list of str
        List of column names to be highlighted.
    color : str, optional
        The background color to be applied for highlighting (default is "yellow").

    Returns:
    --------
    pandas.io.formats.style.Styler
        A Styler object with the specified columns highlighted.
    """

    def highlight(s):
        return [
            f"background-color: {color}" if col in columns else "" for col in s.index
        ]

    return df.style.apply(highlight, axis=1)


################################################################################
############################## Group_by Imputer ################################
################################################################################


def groupby_imputer(
    df: pd.DataFrame,
    impute_col: str,
    by: Union[List[str], str],
    stat: str = "mean",
    fallback: Union[str, float, int] = "global",
    as_new_col: bool = True,
    new_col_name: Optional[str] = None,
):
    """
    Impute missing values in `target` using group-level statistics from `by`.

    - impute_col: column with nulls (e.g. "age")
    - by: one or more categorical columns to group on (e.g. ["sex", "education"])
    - stat: "mean", "median", "min", "max"
    - fallback: what to use if a group has no non-null target
        - "global" -> use overall stat of `target`
        - a number -> use that number
    - as_new_col: if True, don't overwrite; create a new column
    """
    if isinstance(by, str):
        by = [by]

    df_out = df.copy()

    # 1) compute group-level stats
    if stat == "mean":
        grp = df_out.groupby(by)[impute_col].mean()
    elif stat == "median":
        grp = df_out.groupby(by)[impute_col].median()
    elif stat == "min":
        grp = df_out.groupby(by)[impute_col].min()
    elif stat == "max":
        grp = df_out.groupby(by)[impute_col].max()
    else:
        raise ValueError("stat must be one of: mean, median, min, max")

    # 2) attach back to df
    # this gives you a column with group-level value for every row
    df_out["_grp_stat"] = df_out[by].merge(
        grp.rename("_grp_stat"), left_on=by, right_index=True, how="left"
    )["_grp_stat"]

    # 3) compute fallback
    if fallback == "global":
        if stat == "mean":
            fb_val = df_out[impute_col].mean()
        elif stat == "median":
            fb_val = df_out[impute_col].median()
        elif stat == "min":
            fb_val = df_out[impute_col].min()
        elif stat == "max":
            fb_val = df_out[impute_col].max()
    else:
        fb_val = fallback

    # 4) build final series
    if as_new_col:
        out_col = new_col_name or f"{impute_col}_{stat}_imputed"
        df_out[out_col] = df_out[impute_col]
    else:
        out_col = impute_col

    # where target is null → use group stat → if still null → fallback
    df_out.loc[df_out[out_col].isna(), out_col] = df_out.loc[
        df_out[out_col].isna(), "_grp_stat"
    ]
    df_out.loc[df_out[out_col].isna(), out_col] = fb_val

    # clean up helper
    df_out = df_out.drop(columns=["_grp_stat"])

    return df_out


################################################################################
######################### Delete Inactive Dataframes ###########################
################################################################################

# ---------------------------------------------
# optional imports for del_inactive_dataframes
# ---------------------------------------------
try:
    import psutil  # optional, for process memory
except Exception:  # pragma: no cover
    psutil = None

try:
    from rich.console import Console  # type: ignore[import-not-found]
    from rich.table import Table  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    Console = None
    Table = None


def del_inactive_dataframes(
    dfs_to_keep: Union[str, Iterable[str]],
    del_dataframes: bool = False,
    namespace: Optional[Dict[str, Any]] = None,
    include_ipython_cache: bool = False,
    dry_run: bool = False,
    run_gc: bool = True,
    track_memory: bool = False,
    memory_mode: str = "dataframes",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Delete inactive pandas DataFrames from a namespace to help reduce memory usage,
    while keeping only the DataFrames named in `dfs_to_keep`.

    This utility is notebook friendly and cloud friendly:
    - It can optionally include IPython/Jupyter output-cache variables like `_14`, `_15`.
      These can hold references to large DataFrames and keep memory from being reclaimed.
    - It uses Rich for pretty tables when available (Python 3.8+ and rich installed).
      Otherwise it falls back to plain-text output (works on Python 3.7+).
    - It can optionally report memory usage before and after. Process memory reporting uses
      `psutil` if installed. DataFrame memory reporting uses pandas `memory_usage(deep=True)`.

    Parameters
    ----------
    dfs_to_keep : str or Iterable[str]
        Name or names of DataFrame variables to keep. Anything not in this list may be deleted
        if `del_dataframes=True`.

    del_dataframes : bool, default False
        If True, delete DataFrames not listed in `dfs_to_keep`.
        If False, do not delete anything (list and report only).

    namespace : dict or None, default None
        The namespace dictionary to inspect and optionally modify.
        - If None, uses globals() of the module where this function is executed.
        - In notebooks or scripts, passing `namespace=globals()` is often what you want.
        - In other contexts, you can pass `locals()` or a custom dict.

        Mental model:
        - A namespace is simply a dictionary mapping variable names (strings) to objects.
          This function searches that dictionary for values that are pandas DataFrames and,
          if requested, deletes them by removing their names from the dictionary.

        When to use which:
        - Use `globals()` when DataFrames are defined at the notebook or script top level.
        - Use `locals()` when DataFrames are defined inside a function and you want to inspect them there.
        - Use a custom dict when you explicitly manage DataFrames in a container (for example, `frames`).

        Important note about `locals()`:
        - Modifying `locals()` inside a function is not guaranteed to affect actual local variables.
          For reliable memory cleanup inside functions, prefer storing DataFrames in a dict and
          passing that dict as `namespace`.

    include_ipython_cache : bool, default False
        If True, include IPython output-cache names of the form `_<number>` (example: `_14`)
        when searching for and deleting DataFrames.
        If False, these names are ignored.

    dry_run : bool, default False
        If True, compute and display what would be deleted, but do not actually delete anything.

    run_gc : bool, default True
        If True and deletions occur, call `gc.collect()` afterward.

    track_memory : bool, default False
        If True, capture and report memory usage before and after.

    memory_mode : {"dataframes", "all"}, default "dataframes"
        Controls which memory metrics are reported:
        - "dataframes": report only total DataFrame memory using pandas `memory_usage(deep=True)`.
        - "all": also report process RSS if `psutil` is installed.

        Notes:
        - Process RSS is an advisory metric and may not decrease immediately.

    verbose : bool, default True
        If True, print results (Rich tables if available, otherwise plain text).
        If False, do not print anything and only return the summary dictionary.

    Returns
    -------
    dict
        Summary dictionary with:
        - "active": sorted list of DataFrame names found before deletion
        - "to_delete": sorted list of DataFrame names marked for deletion
        - "deleted": sorted list of DataFrame names actually deleted
        - "remaining": sorted list of DataFrame names remaining after deletion
        - "used_rich": bool indicating whether Rich output was used
        - "memory": dict with dataframe memory (and optional RSS) when track_memory=True
    """

    ns = globals() if namespace is None else namespace
    used_rich = Console is not None and Table is not None

    if memory_mode not in {"dataframes", "all"}:
        raise ValueError("memory_mode must be one of: 'dataframes', 'all'")

    # Normalize keep list
    if isinstance(dfs_to_keep, str):
        keep = {dfs_to_keep}
    else:
        keep = set(dfs_to_keep or [])

    def is_ipython_cache_name(name: str) -> bool:
        return name.startswith("_") and name[1:].isdigit()

    def get_process_mem_mb() -> Optional[float]:
        if psutil is None:
            return None
        try:
            proc = psutil.Process(os.getpid())
            return proc.memory_info().rss / (1024**2)
        except Exception:
            return None

    def get_total_df_mem_mb(namespace_dict: Dict[str, Any]) -> float:
        total_bytes = 0
        for name, obj in namespace_dict.items():
            if isinstance(obj, pd.DataFrame):
                if (not include_ipython_cache) and is_ipython_cache_name(name):
                    continue
                try:
                    total_bytes += int(obj.memory_usage(deep=True).sum())
                except Exception:
                    pass
        return total_bytes / (1024**2)

    # Output helpers
    def plain_rule(title: str) -> None:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))

    def plain_list(title: str, names: Iterable[str]) -> None:
        print(title)
        names = list(names)
        if not names:
            print("  (none)")
            return
        for n in names:
            print(f"  - {n}")

    def make_rich_table(title: str, names: Iterable[str], color: str):
        table = Table(title=title)
        table.add_column("DataFrame Name", justify="left", style=color, no_wrap=True)
        for n in names:
            table.add_row(n)
        return table

    def make_rich_memory_table(mem: Dict[str, Any]):
        table = Table(title="Memory Usage (MB)")
        table.add_column("Metric", no_wrap=True)
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Delta", justify="right")

        rows = [("DataFrames total", mem["dataframes_mb"])]

        if mem.get("mode") == "all" and "process_mb" in mem:
            rows.insert(0, ("Process RSS", mem["process_mb"]))

        for label, payload in rows:
            b, a, d = payload["before"], payload["after"], payload["delta"]
            if b is None or a is None or d is None:
                table.add_row(label, "n/a", "n/a", "n/a")
            else:
                table.add_row(label, f"{b:.1f}", f"{a:.1f}", f"{d:+.1f}")

        return table

    def plain_memory_block(mem: Dict[str, Any]) -> None:
        plain_rule("Memory Usage (MB)")

        rows = [("DataFrames total", mem["dataframes_mb"])]

        if mem.get("mode") == "all" and "process_mb" in mem:
            rows.insert(0, ("Process RSS", mem["process_mb"]))

        for label, payload in rows:
            b, a, d = payload["before"], payload["after"], payload["delta"]
            if b is None or a is None or d is None:
                print(f"{label}: (unavailable)")
            else:
                print(f"{label}: {b:.1f} -> {a:.1f} ({d:+.1f})")

    # Memory snapshot before
    want_rss = memory_mode == "all"
    mem_before_proc = get_process_mem_mb() if (track_memory and want_rss) else None
    mem_before_df = get_total_df_mem_mb(ns) if track_memory else None

    # Collect active DataFrames
    active = {}
    for name, obj in ns.items():
        if isinstance(obj, pd.DataFrame):
            if (not include_ipython_cache) and is_ipython_cache_name(name):
                continue
            active[name] = obj
    active_names = sorted(active.keys())

    # Determine deletions
    to_delete = sorted(
        [name for name in active.keys() if del_dataframes and name not in keep]
    )

    # Perform deletions
    deleted = []
    if del_dataframes and (not dry_run):
        for name in to_delete:
            if name in ns:
                ns.pop(name, None)
                deleted.append(name)
        if run_gc and deleted:
            gc.collect()
    deleted = sorted(deleted)

    # Remaining
    remaining = []
    if del_dataframes:
        remaining = sorted(
            name
            for name, obj in ns.items()
            if isinstance(obj, pd.DataFrame)
            and (include_ipython_cache or not is_ipython_cache_name(name))
        )

    # Memory snapshot after
    mem_after_proc = get_process_mem_mb() if (track_memory and want_rss) else None
    mem_after_df = get_total_df_mem_mb(ns) if track_memory else None

    memory = None
    if track_memory:

        def _delta(b, a):
            return None if (b is None or a is None) else (a - b)

        memory = {
            "dataframes_mb": {
                "before": mem_before_df,
                "after": mem_after_df,
                "delta": _delta(mem_before_df, mem_after_df),
            },
            "mode": memory_mode,
        }

        if want_rss:
            memory["process_mb"] = {
                "before": mem_before_proc,
                "after": mem_after_proc,
                "delta": _delta(mem_before_proc, mem_after_proc),
            }

    # Output
    if verbose:
        if used_rich:
            console = Console()

            console.rule("[bold cyan]Active DataFrames[/bold cyan]")
            console.print(
                make_rich_table("Current Active DataFrames", active_names, "magenta")
            )

            if del_dataframes:
                console.rule("[bold yellow]Planned Deletions[/bold yellow]")
                if to_delete:
                    console.print(
                        make_rich_table(
                            "DataFrames Marked for Deletion", to_delete, "yellow"
                        )
                    )
                else:
                    console.print("[dim]No DataFrames match deletion criteria.[/dim]")

                if dry_run:
                    console.print(
                        "[dim]Dry run enabled. No DataFrames were deleted.[/dim]"
                    )
                else:
                    console.rule("[bold red]Deleted DataFrames[/bold red]")
                    if deleted:
                        console.print(
                            make_rich_table("Deleted DataFrames", deleted, "red")
                        )
                    else:
                        console.print("[dim]No DataFrames were deleted.[/dim]")

                console.rule("[bold green]Remaining DataFrames[/bold green]")
                console.print(
                    make_rich_table("Remaining Active DataFrames", remaining, "green")
                )

            if track_memory and memory is not None:
                console.rule("[bold blue]Memory[/bold blue]")
                console.print(make_rich_memory_table(memory))

        else:
            plain_rule("Active DataFrames")
            plain_list("Current Active DataFrames:", active_names)

            if del_dataframes:
                plain_rule("Planned Deletions")
                plain_list("DataFrames Marked for Deletion:", to_delete)

                if dry_run:
                    print("Dry run enabled. No DataFrames were deleted.")
                else:
                    plain_rule("Deleted DataFrames")
                    plain_list("Deleted DataFrames:", deleted)

                plain_rule("Remaining DataFrames")
                plain_list("Remaining Active DataFrames:", remaining)

            if track_memory and memory is not None:
                plain_memory_block(memory)

    return {
        "active": active_names,
        "to_delete": to_delete,
        "deleted": deleted,
        "remaining": remaining if del_dataframes else [],
        "used_rich": used_rich,
        "memory": memory,
    }

################################################################################
############################## Outlier Detection ###############################
################################################################################

def detect_outliers(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
    contamination: float = 0.05,
    return_mask: bool = False,
    return_bounds: bool = False,
    flag_col: Optional[str] = None,
    groupby: Optional[str] = None,
    verbose: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Detect outliers in numeric columns of a DataFrame using IQR, Z-score,
    or Isolation Forest methods.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze for outliers.

    features : list of str, optional
        List of numeric column names to check for outliers. If None, all
        numeric columns are used.

    method : str, optional (default="iqr")
        Outlier detection method. Options are:
        - ``"iqr"``: flags values beyond ``threshold * IQR`` from Q1/Q3.
        - ``"zscore"``: flags values whose absolute Z-score exceeds
          ``threshold``.
        - ``"isoforest"``: uses sklearn's IsolationForest with
          ``contamination`` as the expected outlier fraction.

    threshold : float, optional (default=1.5)
        For ``"iqr"``: the IQR multiplier (e.g. 1.5 for standard Tukey
        fences, 3.0 for extreme outliers).
        For ``"zscore"``: the absolute Z-score cutoff (e.g. 3.0).
        Ignored when ``method="isoforest"``.

    contamination : float, optional (default=0.05)
        Expected proportion of outliers in the dataset. Only used when
        ``method="isoforest"``. Must be between 0 and 0.5.

    return_mask : bool, optional (default=False)
        If True, also returns a boolean DataFrame of the same shape as the
        input (restricted to `features`) where ``True`` indicates an outlier.

    return_bounds : bool, optional (default=False)
        If True, also returns a dictionary mapping each feature name to a
        ``(lower_bound, upper_bound)`` tuple. Bounds are ``"N/A"`` when
        ``method="isoforest"``. The bounds dict can be passed directly to
        ``data_doctor()`` as ``lower_cutoff`` / ``upper_cutoff`` inputs for
        feature-level treatment.

    flag_col : str or None, optional (default=None)
        If provided, adds a boolean column with this name to ``df`` that is
        ``True`` for any row where at least one feature is an outlier.
        The DataFrame is modified in place.

    groupby : str or None, optional (default=None)
        If provided, outlier detection is performed within each group of this
        column rather than across the full dataset. Useful when distributions
        differ meaningfully between groups (e.g. by diagnosis, age group).

    verbose : bool, optional (default=False)
        If True, prints a formatted ASCII summary report to the console
        showing each feature's outlier count, percentage, and bounds —
        similar to the report style used in ``data_doctor()``.

    Returns:
    --------
    summary : pd.DataFrame
        A summary DataFrame with columns:
        ``Variable``, ``Outlier (n)``, ``Outlier (%)``,
        ``Lower Bound``, ``Upper Bound``.
        When ``method="isoforest"``, bounds are shown as ``N/A``.

    mask : pd.DataFrame
        Boolean outlier mask, only returned when ``return_mask=True``.

    bounds : dict
        Dictionary mapping feature names to ``(lower_bound, upper_bound)``
        tuples, only returned when ``return_bounds=True``.

    Raises:
    -------
    ValueError
        If ``method`` is not one of ``"iqr"``, ``"zscore"``, or
        ``"isoforest"``.

    ValueError
        If ``threshold`` is not a positive number.

    ValueError
        If ``contamination`` is not between 0 and 0.5.

    ValueError
        If ``groupby`` column is not found in the DataFrame.

    ValueError
        If none of the specified ``features`` are numeric.

    Notes:
    ------
    - Rows with NaN in a feature are excluded from outlier calculation for
      that feature and are never flagged as outliers.
    - When ``groupby`` is specified, IQR and Z-score bounds are computed
      per group; IsolationForest is fit per group.
    - ``flag_col`` marks a row as an outlier if *any* feature is flagged,
      making it easy to filter the full outlier set with
      ``df[df[flag_col]]``.
    - Use the ``Lower Bound`` and ``Upper Bound`` values from the summary
      or the ``bounds`` dict directly as ``lower_cutoff`` / ``upper_cutoff``
      inputs to ``data_doctor()`` for feature-level transformation and
      treatment.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    valid_methods = ["iqr", "zscore", "isoforest"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid `method` '{method}'. Choose from {valid_methods}."
        )

    if threshold <= 0:
        raise ValueError("`threshold` must be a positive number.")

    if not (0 < contamination <= 0.5):
        raise ValueError("`contamination` must be between 0 and 0.5.")

    if groupby is not None and groupby not in df.columns:
        raise ValueError(
            f"`groupby` column '{groupby}' not found in DataFrame."
        )

    # ------------------------------------------------------------------
    # Resolve features
    # ------------------------------------------------------------------
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()
        if groupby in features:
            features.remove(groupby)
    else:
        features = [
            f for f in features
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        ]

    if not features:
        raise ValueError(
            "No numeric features found. Ensure `features` contains "
            "numeric columns present in the DataFrame."
        )

    # ------------------------------------------------------------------
    # Build boolean outlier mask (same index as df)
    # ------------------------------------------------------------------
    mask = pd.DataFrame(False, index=df.index, columns=features)

    # Store bounds for the summary
    bounds: Dict[str, Tuple] = {}

    if method == "isoforest":
        from sklearn.ensemble import IsolationForest

        if groupby is None:
            clean = df[features].dropna()
            iso = IsolationForest(contamination=contamination, random_state=0)
            preds = iso.fit_predict(clean)
            outlier_idx = clean.index[preds == -1]
            for feat in features:
                mask.loc[outlier_idx, feat] = True
                bounds[feat] = ("N/A", "N/A")
        else:
            for grp, grp_df in df.groupby(groupby, observed=True):
                clean = grp_df[features].dropna()
                if len(clean) < 2:
                    continue
                iso = IsolationForest(contamination=contamination, random_state=0)
                preds = iso.fit_predict(clean)
                outlier_idx = clean.index[preds == -1]
                for feat in features:
                    mask.loc[outlier_idx, feat] = True
            for feat in features:
                bounds[feat] = ("N/A", "N/A")

    else:
        for feat in features:
            series = df[feat].dropna()

            if groupby is None:
                if method == "iqr":
                    low_mask, high_mask, lower, upper = _flag_iqr(series, threshold)
                else:
                    low_mask, high_mask, lower, upper = _flag_zscore(series, threshold)
                outlier_idx = series[low_mask | high_mask].index
                mask.loc[outlier_idx, feat] = True
                bounds[feat] = (round(lower, 4), round(upper, 4))

            else:
                all_lower, all_upper = [], []
                for grp, grp_df in df.groupby(groupby, observed=True):
                    grp_series = grp_df[feat].dropna()
                    if len(grp_series) < 2:
                        continue
                    if method == "iqr":
                        low_m, high_m, lower, upper = _flag_iqr(grp_series, threshold)
                    else:
                        low_m, high_m, lower, upper = _flag_zscore(grp_series, threshold)
                    outlier_idx = grp_series[low_m | high_m].index
                    mask.loc[outlier_idx, feat] = True
                    all_lower.append(lower)
                    all_upper.append(upper)
                bounds[feat] = (
                    round(min(all_lower), 4) if all_lower else np.nan,
                    round(max(all_upper), 4) if all_upper else np.nan,
                )

    # ------------------------------------------------------------------
    # Add flag column to df if requested
    # ------------------------------------------------------------------
    if flag_col is not None:
        df[flag_col] = mask.any(axis=1)

    # ------------------------------------------------------------------
    # Build summary DataFrame
    # ------------------------------------------------------------------
    total = len(df)
    summary_rows = []
    for feat in features:
        n_outliers = mask[feat].sum()
        pct = round(100 * n_outliers / total, 2)
        lower_b, upper_b = bounds[feat]
        summary_rows.append(
            {
                "Variable": feat,
                "Outlier (n)": int(n_outliers),
                "Outlier (%)": pct,
                "Lower Bound": lower_b,
                "Upper Bound": upper_b,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        "Outlier (%)", ascending=False
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Verbose ASCII report
    # ------------------------------------------------------------------
    if verbose:
        method_label = {
            "iqr": f"IQR (threshold={threshold})",
            "zscore": f"Z-Score (threshold={threshold})",
            "isoforest": (
                f"Isolation Forest "
                f"(contamination={contamination})"
            ),
        }[method]

        # Column widths — fixed so header and data rows always align
        c0, c1, c2, c3, c4 = 22, 12, 12, 14, 14
        # Top info table value col spans remaining cols + separators
        info_key_w = c0
        info_val_w = c1 + c2 + c3 + c4 + 9
        total_width = c0 + c1 + c2 + c3 + c4 + 14

        sep = (
            f"+{'-'*(c0+2)}+{'-'*(c1+2)}"
            f"+{'-'*(c2+2)}+{'-'*(c3+2)}+{'-'*(c4+2)}+"
        )

        print(
            "OUTLIER DETECTION SUMMARY REPORT".center(total_width)
        )
        print(f"+{'-'*(info_key_w+2)}+{'-'*(info_val_w+2)}+")
        print(
            f"| {'Method':<{info_key_w}} "
            f"| {method_label:<{info_val_w}} |"
        )
        print(
            f"| {'Total rows':<{info_key_w}} "
            f"| {total:<{info_val_w},} |"
        )
        if groupby:
            print(
                f"| {'Grouped by':<{info_key_w}} "
                f"| {groupby:<{info_val_w}} |"
            )
        print(sep)
        print(
            f"| {'Variable':<{c0}} | {'Outlier (n)':>{c1}} | "
            f"{'Outlier (%)':>{c2}} | "
            f"{'Lower Bound':>{c3}} | {'Upper Bound':>{c4}} |"
        )
        print(sep)
        for _, row in summary.iterrows():
            lb = (
                f"{row['Lower Bound']}"
                if row['Lower Bound'] == "N/A"
                else f"{row['Lower Bound']:,.4f}"
            )
            ub = (
                f"{row['Upper Bound']}"
                if row['Upper Bound'] == "N/A"
                else f"{row['Upper Bound']:,.4f}"
            )
            print(
                f"| {str(row['Variable']):<{c0}} "
                f"| {row['Outlier (n)']:>{c1},} | "
                f"{row['Outlier (%)']:>{c2}.2f} | "
                f"{lb:>{c3}} | {ub:>{c4}} |"
            )
        print(sep)
        total_flagged = mask.any(axis=1).sum()
        print(
            f"\nTotal flagged rows (any feature): "
            f"{total_flagged:,} "
            f"({100 * total_flagged / total:.2f}%)"
        )

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    returns = [summary]
    if return_mask:
        returns.append(mask)
    if return_bounds:
        returns.append(bounds)

    # When verbose=True and no structured returns are requested, suppress
    # auto-display in Jupyter by returning None — the ASCII report is sufficient
    if verbose and not (return_mask or return_bounds):
        return None

    if len(returns) == 1:
        return returns[0]
    return tuple(returns)

################################################################################
############################## Normality Tests #################################
################################################################################

def normality_tests(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    alpha: float = 0.05,
    tests: Optional[List[str]] = None,
    decimal_places: int = 6,
) -> pd.DataFrame:
    """
    Run batch normality tests across numeric columns of a DataFrame.

    Applies Shapiro-Wilk, D'Agostino K², and/or Anderson-Darling tests
    to each specified feature and returns a summary DataFrame with test
    statistics, p-values, and a pass/fail determination at the given
    significance level.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze.

    features : list of str, optional
        List of numeric column names to test. If ``None``, all numeric
        columns are used.

    alpha : float, optional (default=0.05)
        Significance level for the pass/fail determination. A feature
        is considered normally distributed (``True``) if the p-value
        exceeds ``alpha``. For Anderson-Darling, the test statistic is
        compared to the critical value at the significance level closest
        to ``alpha``.

    tests : list of str, optional
        List of tests to run. Options are ``"shapiro"``,
        ``"dagostino"``, and ``"anderson"``. Defaults to all three if
        not specified.

    decimal_places : int, optional (default=6)
        Number of decimal places to round the ``Statistic`` and
        ``P-value`` columns in the returned summary DataFrame.

    Returns:
    --------
    pd.DataFrame
        A summary DataFrame with columns:
        ``Variable``, ``Test``, ``Statistic``, ``P-value``,
        ``Normal``.

        - ``P-value`` for Anderson-Darling is interpolated from
          pre-calculated tables when scipy >= 1.17 is installed.
          On older scipy versions, ``"-"`` is shown and pass/fail is
          determined by comparing the statistic against the critical
          value at ``alpha``.
        - ``Normal`` is ``True`` if the feature passes the normality
          test at the given ``alpha``, ``False`` otherwise.

    Raises:
    -------
    ValueError
        If ``alpha`` is not between 0 and 1.

    ValueError
        If any value in ``tests`` is not one of ``"shapiro"``,
        ``"dagostino"``, or ``"anderson"``.

    ValueError
        If no numeric features are found.

    Notes:
    ------
    - Shapiro-Wilk is most reliable for small samples (n < 5,000).
      For larger samples it tends to reject normality even for
      distributions that are approximately normal.
    - D'Agostino K² is better suited for larger samples and tests
      for skewness and kurtosis jointly.
    - Anderson-Darling does not produce a p-value. Pass/fail is
      determined by comparing the test statistic against the critical
      value at the significance level closest to ``alpha`` from the
      set ``[15%, 10%, 5%, 2.5%, 1%]``.
    - NaN values are dropped per feature before testing.
    - Results are sorted by ``Variable`` then ``Test`` for readability.
    """

    from scipy import stats as _stats

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    valid_tests = ["shapiro", "dagostino", "anderson"]

    if tests is None:
        tests = valid_tests

    invalid = [t for t in tests if t not in valid_tests]
    if invalid:
        raise ValueError(
            f"Invalid test(s) {invalid}. "
            f"Choose from {valid_tests}."
        )

    if not (0 < alpha < 1):
        raise ValueError("`alpha` must be between 0 and 1.")

    # ------------------------------------------------------------------
    # Resolve features
    # ------------------------------------------------------------------
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()
    else:
        features = [
            f for f in features
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        ]

    if not features:
        raise ValueError(
            "No numeric features found. Ensure `features` contains "
            "numeric columns present in the DataFrame."
        )

    # Anderson-Darling significance levels available from scipy
    _anderson_sig_levels = [0.15, 0.10, 0.05, 0.025, 0.01]

    def _closest_anderson_idx(alpha):
        """Return index of the significance level closest to alpha."""
        return min(
            range(len(_anderson_sig_levels)),
            key=lambda i: abs(_anderson_sig_levels[i] - alpha),
        )

    # ------------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------------
    rows = []

    for feat in features:
        series = df[feat].dropna().values

        if "shapiro" in tests:
            if len(series) < 3:
                rows.append({
                    "Variable": feat,
                    "Test": "Shapiro-Wilk",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Normal": np.nan,
                })
            elif len(series) > 5000:
                # Shapiro-Wilk is unreliable for n > 5,000 — skip it
                rows.append({
                    "Variable": feat,
                    "Test": "Shapiro-Wilk",
                    "Statistic": "-",
                    "P-value": "-",
                    "Normal": "-",
                })
            else:
                stat, p = _stats.shapiro(series)
                normal = bool(p > alpha)
                rows.append({
                    "Variable": feat,
                    "Test": "Shapiro-Wilk",
                    "Statistic": round(stat, decimal_places),
                    "P-value": round(p, decimal_places),
                    "Normal": normal,
                })

        if "dagostino" in tests:
            if len(series) < 8:
                stat, p, normal = np.nan, np.nan, np.nan
            else:
                stat, p = _stats.normaltest(series)
                normal = bool(p > alpha)
            rows.append({
                "Variable": feat,
                "Test": "D'Agostino K²",
                "Statistic": round(stat, decimal_places) if not np.isnan(stat) else np.nan,
                "P-value": round(p, decimal_places) if not np.isnan(p) else np.nan,
                "Normal": normal,
            })

        if "anderson" in tests:
            if len(series) < 3:
                rows.append({
                    "Variable": feat,
                    "Test": "Anderson-Darling",
                    "Statistic": np.nan,
                    "P-value": "-",
                    "Normal": "-",
                })
            else:
                # scipy >= 1.17 supports method="interpolate" which
                # returns a p-value and silences the FutureWarning.
                # Fall back to critical value comparison on older scipy.
                try:
                    result = _stats.anderson(
                        series,
                        dist="norm",
                        method="interpolate",
                    )
                    p_ad = round(float(result.pvalue), decimal_places)
                    normal = bool(p_ad > alpha)
                except TypeError:
                    result = _stats.anderson(series, dist="norm")
                    idx = _closest_anderson_idx(alpha)
                    critical_val = result.critical_values[idx]
                    normal = bool(result.statistic < critical_val)
                    p_ad = "-"
                rows.append({
                    "Variable": feat,
                    "Test": "Anderson-Darling",
                    "Statistic": round(
                        result.statistic, decimal_places
                    ),
                    "P-value": p_ad,
                    "Normal": normal,
                })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["Variable", "Test"])
        .reset_index(drop=True)
    )

    # Notify user if any features were too large for Shapiro-Wilk
    shapiro_skipped = summary[
        (summary["Test"] == "Shapiro-Wilk") & (summary["Statistic"] == "-")
    ]["Variable"].tolist()
    if shapiro_skipped:
        skipped_str = ", ".join(shapiro_skipped)
        print(
            f"\nNote: Shapiro-Wilk was skipped for: {skipped_str}"
        )
        print("Reason: n > 5,000. The test is unreliable at large n")
        print("— even trivial deviations cause rejection.")
        print("Use D'Agostino K\u00b2 or Anderson-Darling instead.")
        print()

    return summary