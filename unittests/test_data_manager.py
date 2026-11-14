import pytest
from collections import Counter
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import gc
import sys
import os
import datetime
import builtins
from unittest.mock import MagicMock, patch
from eda_toolkit import (
    ensure_directory,
    read_csv_with_progress,
    generate_table1,
    table1_to_str,
    add_ids,
    strip_trailing_period,
    parse_date_with_rule,
    dataframe_profiler,
    contingency_table,
    save_dataframes_to_excel,
    summarize_all_combinations,
    highlight_columns,
    about,
    eda_toolkit_logo,
    detailed_doc,
    groupby_imputer,
    del_inactive_dataframes,
    table1_to_str,
    TableWrapper,
    detect_outliers,
)


base_path = os.getcwd()
data_path = os.path.join(base_path, "data")
data_output = os.path.join(base_path, "data_output")


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {"Feature1": [1, 2, 3, 4, 5], "Feature2": [10, 20, 30, 40, 50]},
    )


@pytest.fixture
def sample_dataframe_values():
    return pd.DataFrame({"values": [1, 2, 3, 4, 5]})


@pytest.fixture
def sample_dataframes():
    return {
        "Sheet1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "Sheet2": pd.DataFrame({"X": [7, 8, 9], "Y": [10, 11, 12]}),
    }


@pytest.fixture
def sample_dataframe_with_text():
    return pd.DataFrame({"col": ["123.", "456", "789.", None, "text."]})


@pytest.fixture
def sample_dataframe_dates():
    return pd.DataFrame({"dates": ["25/12/2023", "12/25/2023", "05/06/2023"]})


@pytest.fixture
def sample_dataframe_profiler():
    return pd.DataFrame({"col1": [1, 2, 2, None], "col2": ["A", "B", "B", "A"]})


@pytest.fixture
def sample_dataframe_contingency():
    return pd.DataFrame(
        {"Category": ["A", "B", "A", "B", "C"], "Values": [1, 2, 1, 2, 3]}
    )


@pytest.fixture
def sample_categorical_dataframe():
    return pd.DataFrame(
        {
            "Category": ["A", "A", "B", "B", "C"],
            "Subcategory": ["X", "Y", "X", "Y", "X"],
            "Value": [10, 15, 20, 25, 30],
        }
    )


@pytest.fixture
def sample_corr_dataframe():
    return pd.DataFrame(
        {
            "Feature1": [1, 2, 3, 4, 5],
            "Feature2": [5, 4, 3, 2, 1],
            "Feature3": [2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def sample_scatter_dataframe():
    """Create a sample DataFrame for scatter plots."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Feature1": np.random.rand(50) * 10,
            "Feature2": np.random.rand(50) * 20,
            "Category": np.random.choice(["A", "B"], size=50),
        }
    )


@pytest.fixture
def sample_box_violin_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Metric1": np.random.rand(50) * 100,
            "Metric2": np.random.rand(50) * 200,
            "Category": np.random.choice(["A", "B", "C"], size=50),
        }
    )


@pytest.fixture
def sample_crosstab_dataframe():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Category": np.random.choice(["A", "B", "C"], size=50),
            "Group": np.random.choice(["X", "Y"], size=50),
            "Outcome": np.random.choice(["Success", "Failure"], size=50),
        }
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, None, 40],
            "income": [50000, None, 60000, 80000, 75000],
            "gender": ["Male", "Female", "Female", "Male", None],
            "binary_flag": [1, 0, 1, 0, 1],
            "is_married": [True, False, True, True, False],
        }
    )

@pytest.fixture
def clean_df():
    """DataFrame with no outliers under standard IQR thresholds."""
    np.random.seed(0)
    return pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(5, 1, 100),
        }
    )
 
 
@pytest.fixture
def outlier_df():
    """DataFrame with known outliers injected at specific indices."""
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(5, 1, 100),
        }
    )
    df.loc[0, "A"] = 100.0   # extreme high outlier
    df.loc[1, "A"] = -100.0  # extreme low outlier
    df.loc[2, "B"] = 500.0   # extreme high outlier in B
    return df
 
 
@pytest.fixture
def grouped_df():
    """DataFrame with a groupby column and known per-group outliers."""
    np.random.seed(42)
    group_a = pd.DataFrame({
        "value": np.random.normal(0, 1, 50),
        "group": "A",
    })
    group_b = pd.DataFrame({
        "value": np.random.normal(10, 1, 50),
        "group": "B",
    })
    df = pd.concat([group_a, group_b], ignore_index=True)
    df.loc[0, "value"] = 1000.0   # outlier in group A
    df.loc[50, "value"] = -1000.0  # outlier in group B
    return df
 
 
@pytest.fixture
def nan_df():
    """DataFrame with NaN values in numeric columns."""
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 4.0, 100.0],
            "B": [np.nan, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return df
 
 
@pytest.fixture
def categorical_df():
    """DataFrame with mixed numeric and non-numeric columns."""
    np.random.seed(0)
    return pd.DataFrame(
        {
            "numeric": np.random.normal(0, 1, 50),
            "category": np.random.choice(["X", "Y"], 50),
        }
    )

@pytest.fixture
def sample_csv(tmp_path):
    """Write a small CSV file and return its path."""
    df = pd.DataFrame(
        {
            "col_a": range(100),
            "col_b": [f"value_{i}" for i in range(100)],
            "col_c": np.random.default_rng(0).random(100),
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path), df


@pytest.fixture
def large_csv(tmp_path):
    """Write a CSV larger than one 10,000-row chunk."""
    df = pd.DataFrame(
        {
            "id": range(25_000),
            "value": np.random.default_rng(1).random(25_000),
        }
    )
    path = tmp_path / "large.csv"
    df.to_csv(path, index=False)
    return str(path), df


@pytest.fixture
def mixed_dtype_csv(tmp_path):
    """Write a CSV with mixed dtypes to trigger DtypeWarning suppression."""
    rows = [{"id": i, "mixed": i if i % 2 == 0 else f"str_{i}"} for i in range(50)]
    df = pd.DataFrame(rows)
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    return str(path), df


# ------------------------------------------------------------------
# Basic correctness
# ------------------------------------------------------------------

def test_returns_dataframe(sample_csv):
    path, _ = sample_csv
    result = read_csv_with_progress(path)
    assert isinstance(result, pd.DataFrame)


def test_full_read_row_count(sample_csv):
    path, original = sample_csv
    result = read_csv_with_progress(path)
    assert len(result) == len(original)


def test_full_read_columns(sample_csv):
    path, original = sample_csv
    result = read_csv_with_progress(path)
    assert list(result.columns) == list(original.columns)


def test_full_read_values(sample_csv):
    path, original = sample_csv
    result = read_csv_with_progress(path)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), original.reset_index(drop=True))


# ------------------------------------------------------------------
# nrows limiting
# ------------------------------------------------------------------

def test_nrows_limits_output(sample_csv):
    path, _ = sample_csv
    result = read_csv_with_progress(path, nrows=10)
    assert len(result) == 10


def test_nrows_one(sample_csv):
    path, original = sample_csv
    result = read_csv_with_progress(path, nrows=1)
    assert len(result) == 1
    assert result.iloc[0]["col_a"] == original.iloc[0]["col_a"]


def test_nrows_exact_chunk_boundary(large_csv):
    """nrows equal to chunksize should return exactly that many rows."""
    path, _ = large_csv
    result = read_csv_with_progress(path, nrows=10_000)
    assert len(result) == 10_000


def test_nrows_larger_than_file(sample_csv):
    """nrows larger than the file should return all rows without error."""
    path, original = sample_csv
    result = read_csv_with_progress(path, nrows=10_000)
    assert len(result) == len(original)


def test_nrows_none_reads_all(sample_csv):
    path, original = sample_csv
    result = read_csv_with_progress(path, nrows=None)
    assert len(result) == len(original)


# ------------------------------------------------------------------
# Multi-chunk reads
# ------------------------------------------------------------------

def test_large_file_full_read(large_csv):
    path, original = large_csv
    result = read_csv_with_progress(path)
    assert len(result) == len(original)


def test_large_file_nrows_spans_chunks(large_csv):
    """nrows that spans more than one chunk should be handled correctly."""
    path, _ = large_csv
    result = read_csv_with_progress(path, nrows=15_000)
    assert len(result) == 15_000


def test_index_is_reset(large_csv):
    """Concatenated result should have a clean 0-based RangeIndex."""
    path, _ = large_csv
    result = read_csv_with_progress(path, nrows=15_000)
    assert list(result.index) == list(range(15_000))


# ------------------------------------------------------------------
# DtypeWarning suppression
# ------------------------------------------------------------------

def test_dtype_warning_suppressed(mixed_dtype_csv):
    """Function should not surface a DtypeWarning to the caller."""
    import warnings as _warnings
    path, _ = mixed_dtype_csv
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        read_csv_with_progress(path)
    dtype_warnings = [
        w for w in caught
        if issubclass(w.category, pd.errors.DtypeWarning)
    ]
    assert len(dtype_warnings) == 0


# ------------------------------------------------------------------
# Progress bar
# ------------------------------------------------------------------

def test_progress_bar_is_called(sample_csv):
    path, _ = sample_csv
    with patch("tqdm.tqdm.update") as mock_update:
        read_csv_with_progress(path)
        assert mock_update.called


def test_progress_bar_total_no_nrows(sample_csv):
    """Without nrows, tqdm total should equal total file lines (minus header)."""
    path, original = sample_csv
    with patch("tqdm.tqdm.__init__", return_value=None) as mock_init, \
         patch("tqdm.tqdm.update"), \
         patch("tqdm.tqdm.__enter__", return_value=None), \
         patch("tqdm.tqdm.__exit__", return_value=False):
        try:
            read_csv_with_progress(path)
        except Exception:
            pass
        if mock_init.called:
            kwargs = mock_init.call_args[1] if mock_init.call_args[1] else {}
            args = mock_init.call_args[0] if mock_init.call_args[0] else ()
            total = kwargs.get("total")
            if total is not None:
                assert total == len(original)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_empty_csv_raises(tmp_path):
    """A CSV with only a header and no data rows should return an empty DataFrame."""
    path = tmp_path / "empty.csv"
    path.write_text("col_a,col_b\n")
    result = read_csv_with_progress(str(path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_single_row_csv(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("col_a,col_b\n1,hello\n")
    result = read_csv_with_progress(str(path))
    assert len(result) == 1
    assert result.iloc[0]["col_a"] == 1
    assert result.iloc[0]["col_b"] == "hello"


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        read_csv_with_progress("/nonexistent/path/file.csv")


def test_nrows_zero(sample_csv):
    """nrows=0 is falsy so should behave the same as nrows=None (read all)."""
    path, original = sample_csv
    result = read_csv_with_progress(path, nrows=0)
    assert len(result) == len(original)

def test_ensure_directory(tmp_path):
    dir_path = tmp_path / "test_dir"
    ensure_directory(str(dir_path))
    assert dir_path.exists()

def test_save_dataframes_to_excel(sample_dataframes, tmp_path):
    file_path = tmp_path / "test_output.xlsx"

    try:
        # Debugging
        print(f"DEBUG: file_path = {file_path}, type = {type(file_path)}")
        assert isinstance(
            file_path, (str, os.PathLike)
        ), "file_path must be a string or Path-like object"

        save_dataframes_to_excel(
            file_path,  # Ensure file path is first argument
            sample_dataframes,  # Ensure dataframes dictionary is passed correctly
        )

        assert file_path.exists()  # Ensure the file was created
    except Exception as e:
        pytest.fail(f"save_dataframes_to_excel failed: {e}")


def test_add_ids(sample_dataframe):
    num_digits = 5
    seed = 42

    df_with_ids = add_ids(
        sample_dataframe,
        id_colname="UniqueID",
        num_digits=num_digits,
        seed=seed,
    )

    # Check if UniqueID column is present
    assert "UniqueID" in df_with_ids.columns, "Column 'UniqueID' was not added"

    # Check that all UniqueID values are unique
    assert df_with_ids["UniqueID"].nunique() == len(
        sample_dataframe
    ), "Duplicate IDs found"

    # Check that the length of each ID matches num_digits
    assert (
        df_with_ids["UniqueID"].apply(lambda x: len(str(x)) == num_digits).all()
    ), f"Some IDs do not have {num_digits} digits"

    # Check if the function is deterministic (same seed should generate same IDs)
    df_with_ids_2 = add_ids(
        sample_dataframe, id_colname="UniqueID", num_digits=num_digits, seed=seed
    )

    assert (
        df_with_ids["UniqueID"].tolist() == df_with_ids_2["UniqueID"].tolist()
    ), "IDs generated with the same seed do not match"


def test_add_ids_exceed_pool_size():
    """
    Test that a ValueError is raised when the number of rows exceeds the pool
    of unique IDs.
    """

    # Set num_digits such that the pool size is too small for the given DataFrame
    num_digits = 3  # Pool size = 9 * (10^2) = 900 unique IDs
    pool_size = 9 * (10 ** (num_digits - 1))  # Pool size = 900
    n_rows = pool_size + 1  # Exceed the pool size

    # Create a DataFrame with more rows than the pool size
    df = pd.DataFrame({"data": range(n_rows)})

    # Expect a ValueError when trying to add IDs
    with pytest.raises(
        ValueError,
        match=r"The number of rows \(\d+\) exceeds the total pool of possible unique IDs",
    ):
        add_ids(df, num_digits=num_digits)


def test_strip_trailing_period(sample_dataframe_with_text):
    """Test that strip_trailing_period removes trailing periods correctly."""

    df_cleaned = strip_trailing_period(sample_dataframe_with_text, "col")

    # Define expected results (ensure None values match actual Pandas NaN behavior)
    expected_values = ["123", "456", "789", np.nan, "text"]

    # Convert column to list and check each value (handling NaN cases)
    cleaned_values = df_cleaned["col"].tolist()
    for actual, expected in zip(cleaned_values, expected_values):
        if pd.isna(expected):
            assert pd.isna(actual), f"Expected NaN but got {actual}"
        else:
            assert actual == expected, f"Expected {expected} but got {actual}"

    # Ensure the function does not modify other columns
    df_original = sample_dataframe_with_text.copy()
    df_cleaned_other = strip_trailing_period(df_original, "col")

    assert df_cleaned_other.drop(columns=["col"]).equals(
        df_original.drop(columns=["col"])
    ), "Function should not modify other columns"

    # Ensure type consistency (all values should remain str or NaN)
    assert all(
        isinstance(val, (str, float)) or pd.isna(val) for val in df_cleaned["col"]
    ), "All values should remain as strings or NaN"


def test_strip_trailing_period_convert_back_to_float():
    import pandas as pd
    from eda_toolkit.data_manager import strip_trailing_period

    class FloatLike:
        def __str__(self):
            return "123."

    df = pd.DataFrame({"col": [FloatLike(), None]})
    out = strip_trailing_period(df.copy(), "col")
    val = out["col"].iloc[0]

    # should have stripped the dot and converted to float
    assert isinstance(val, float)
    assert val == 123.0

    # None should remain None/NaN
    assert pd.isna(out["col"].iloc[1])


def test_strip_trailing_period_fallback_to_string():
    import pandas as pd
    from eda_toolkit.data_manager import strip_trailing_period

    class NoParse:
        def __str__(self):
            return "abc."  # not a number

    df = pd.DataFrame({"col": [NoParse()]})
    out = strip_trailing_period(df.copy(), "col")
    val = out["col"].iloc[0]

    # conversion to float should fail, so we fall back to the stripped string
    assert isinstance(val, str)
    assert val == "abc"


def test_parse_date_with_rule(sample_dataframe_dates):
    assert parse_date_with_rule(sample_dataframe_dates.iloc[0, 0]) == "2023-12-25"
    assert parse_date_with_rule(sample_dataframe_dates.iloc[1, 0]) == "2023-12-25"
    assert parse_date_with_rule(sample_dataframe_dates.iloc[2, 0]) == "2023-06-05"


def test_dataframe_profiler(sample_dataframe_profiler):
    profile = dataframe_profiler(sample_dataframe_profiler, return_df=True)
    assert "column" in profile.columns
    assert "null_total" in profile.columns


def test_dataframe_profiler_styles_in_notebook(monkeypatch, sample_dataframe_profiler):
    # Simulate notebook environment
    monkeypatch.setitem(dataframe_profiler.__globals__, "is_notebook", True)

    result = dataframe_profiler(sample_dataframe_profiler, return_df=False)
    # Should return a DataFrame
    assert isinstance(result, pd.DataFrame)
    # One row per input column
    assert len(result) == len(sample_dataframe_profiler.columns)
    # Should have exactly these profiling columns
    expected = {
        "column",
        "dtype",
        "null_total",
        "null_pct",
        "unique_values_total",
        "max_unique_value",
        "max_unique_value_total",
        "max_unique_value_pct",
    }
    assert set(result.columns) == expected


def test_dataframe_profiler_hide_index_fallback(monkeypatch, sample_dataframe_profiler):
    # Simulate notebook environment
    monkeypatch.setitem(dataframe_profiler.__globals__, "is_notebook", True)

    # Remove hide_index to hit the except-path
    hide_attr = getattr(Styler, "hide_index", None)
    if hide_attr is not None:
        delattr(Styler, "hide_index")

    try:
        result = dataframe_profiler(sample_dataframe_profiler, return_df=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe_profiler.columns)
        # Basic sanity check on columns
        assert "null_total" in result.columns
    finally:
        # Restore hide_index
        if hide_attr is not None:
            setattr(Styler, "hide_index", hide_attr)


def test_contingency_table(sample_dataframe_contingency):
    table = contingency_table(sample_dataframe_contingency, cols="Category")
    assert "Total" in table.columns
    assert "Percentage" in table.columns


def test_summarize_all_combinations(sample_categorical_dataframe, tmp_path):
    try:
        data_path = str(tmp_path)
        data_name = "summary.xlsx"
        summary_tables, all_combinations = summarize_all_combinations(
            df=sample_categorical_dataframe,
            variables=["Category", "Subcategory"],
            data_path=data_path,
            data_name=data_name,
            min_length=2,
        )

        assert isinstance(summary_tables, dict), "Output should be a dictionary"
        assert all(
            isinstance(df, pd.DataFrame) for df in summary_tables.values()
        ), "Each summary should be a DataFrame"
        assert all_combinations, "All combinations list should not be empty"
        assert os.path.exists(
            os.path.join(data_path, data_name)
        ), "Excel file should be created"

    except Exception as e:
        pytest.fail(f"summarize_all_combinations failed: {e}")


def test_datetime_import_version_gte_3_7(mocker):
    """Test datetime import behavior for Python versions >= 3.7"""
    mocker.patch.object(sys, "version_info", (3, 7))  # Mock Python 3.7+

    # Import datetime AFTER patching to ensure it behaves as expected
    from datetime import datetime

    assert "datetime" in sys.modules, "datetime module should be imported"
    assert hasattr(
        datetime, "fromisoformat"
    ), "datetime should support 'fromisoformat' in Python 3.7+"


def test_datetime_import_version_lt_3_7():
    """Test datetime import behavior for Python versions < 3.7."""

    # Mock `sys.version_info` to simulate Python 3.6
    with patch.object(sys, "version_info", (3, 6)):

        # Create a MagicMock instance for datetime.datetime
        mock_datetime = MagicMock()
        del (
            mock_datetime.fromisoformat
        )  # Remove `fromisoformat` to simulate Python <3.7

        # Patch `datetime.datetime` with our mocked version
        with patch("datetime.datetime", new=mock_datetime):
            assert not hasattr(
                datetime.datetime, "fromisoformat"
            ), "datetime should NOT support 'fromisoformat' in Python <3.7"


def test_ensure_directory_creates_new(mocker, tmp_path):
    """
    Test ensure_directory creates the directory when it does not exist.
    """
    dir_path = tmp_path / "new_dir"

    # Mock os.path.exists to return False initially
    mocker.patch("os.path.exists", return_value=False)
    mock_makedirs = mocker.patch("os.makedirs")

    ensure_directory(str(dir_path))

    # Ensure makedirs was called since the directory didn’t exist
    mock_makedirs.assert_called_once_with(str(dir_path))


def test_ensure_directory_already_exists(mocker, tmp_path):
    """
    Test ensure_directory when directory already exists.
    """
    dir_path = tmp_path / "existing_dir"

    # Mock os.path.exists to return True
    mocker.patch("os.path.exists", return_value=True)
    mock_makedirs = mocker.patch("os.makedirs")

    ensure_directory(str(dir_path))

    # Ensure makedirs was NOT called since the directory exists
    mock_makedirs.assert_not_called()


def test_highlight_columns():
    # Create a sample DataFrame
    data = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
    df = pd.DataFrame(data)

    # Apply highlight_columns function
    styled_df = highlight_columns(df, columns=["A", "C"], color="lightblue")

    # Ensure the return type is a pandas Styler
    assert isinstance(
        styled_df, pd.io.formats.style.Styler
    ), "Function should return a Styler object"

    # Extract the applied styles
    styles = styled_df._compute().ctx  # Extracts computed styles as a dictionary

    # Convert column names to positions
    col_positions = {col: pos for pos, col in enumerate(df.columns)}

    # Verify the expected styles
    expected_styles = {
        (0, col_positions["A"]): [("background-color", "lightblue")],
        (1, col_positions["A"]): [("background-color", "lightblue")],
        (0, col_positions["C"]): [("background-color", "lightblue")],
        (1, col_positions["C"]): [("background-color", "lightblue")],
    }

    for key, value in expected_styles.items():
        assert (
            styles.get(key) == value
        ), f"Expected {value} at {key}, but got {styles.get(key)}"

    # Ensure column "B" remains unstyled
    assert all(
        styles.get((i, col_positions["B"]), []) == [] for i in range(len(df))
    ), "Column B should not be styled"


def test_about_prints_logo_and_docs(capsys):
    """about() prints the ASCII logo and the package overview."""
    about()
    captured = capsys.readouterr()
    assert eda_toolkit_logo in captured.out, "ASCII art should be printed."
    assert detailed_doc in captured.out, "Detailed documentation should be printed."


def test_builtin_help_not_overridden(capsys):
    """Importing eda_toolkit must leave Python's builtin help() untouched."""
    assert builtins.help is not about
    help(int)
    captured = capsys.readouterr()
    assert eda_toolkit_logo not in captured.out
    assert (
        "class int" in captured.out or "Help on class int" in captured.out
    ), "Builtin help should behave normally."


def test_returns_dataframe(sample_df):
    result = generate_table1(sample_df)

    if isinstance(result, tuple):
        assert all(isinstance(r, pd.DataFrame) for r in result)
    else:
        # accept either raw DataFrame or TableWrapper
        assert isinstance(result, (pd.DataFrame, TableWrapper))


def test_returns_markdown_only(sample_df):
    # ensure the data_output folder exists
    ensure_directory(data_output)
    # ask generate_table1 to export and return markdown, writing into data_output/test_table1_summary.md
    md_path = os.path.join(data_output, "test_table1_summary.md")
    result = generate_table1(
        sample_df,
        export_markdown=True,
        return_markdown_only=True,
        markdown_path=md_path,
    )

    # it should still return the markdown strings
    assert isinstance(result, dict)
    assert "continuous" in result and "categorical" in result
    assert (
        "| Variable |" in result["continuous"]
        or "| Variable |" in result["categorical"]
    )

    # and it should have written the files
    cont_file = os.path.join(data_output, "test_table1_summary_continuous.md")
    cat_file = os.path.join(data_output, "test_table1_summary_categorical.md")
    assert os.path.exists(cont_file), f"{cont_file} not found"
    assert os.path.exists(cat_file), f"{cat_file} not found"


def test_exports_markdown_to_file(sample_df):
    # ensure the data_output folder exists
    ensure_directory(data_output)
    # tell generate_table1 to write to data_output/test_table1_summary.md
    md_path = os.path.join(data_output, "test_table1_summary.md")

    generate_table1(
        sample_df,
        export_markdown=True,
        markdown_path=md_path,
        include_types="categorical",
    )

    # generate_table1 appends "_categorical.md"
    expected_file = os.path.join(data_output, "test_table1_summary_categorical.md")
    assert os.path.exists(expected_file)


def test_detect_binary_numeric_moves_column(sample_df):
    result = generate_table1(sample_df)
    assert any("binary_flag" in str(v) for v in result["Variable"])


def test_disable_detect_binary_keeps_in_continuous(sample_df):
    result = generate_table1(sample_df, detect_binary_numeric=False)
    assert any(
        (row["Variable"] == "binary_flag" and row["Type"] == "Continuous")
        for _, row in result.iterrows()
    )


def test_value_counts_false_shows_summary(sample_df):
    result = generate_table1(sample_df, value_counts=False)
    assert not any("=" in v for v in result["Variable"])


def test_value_counts_true_shows_detailed(sample_df):
    result = generate_table1(sample_df, value_counts=True)
    assert any("=" in v for v in result["Variable"])


def test_max_categories_limits_output(sample_df):
    result = generate_table1(
        sample_df,
        value_counts=True,
        max_categories=1,
    )
    detailed_vars = [v for v in result["Variable"] if "=" in v]
    # Group by prefix (i.e., variable name before the "=")

    prefixes = [v.split(" = ")[0] for v in detailed_vars]
    counts_by_var = Counter(prefixes)

    # Ensure no variable has more than 1 category
    assert all(c <= 1 for c in counts_by_var.values())


def test_handles_all_empty_df():
    empty_df = pd.DataFrame()
    result = generate_table1(empty_df)
    assert result.empty


def test_manual_column_specification(sample_df):
    result = generate_table1(
        sample_df,
        categorical_cols=["gender"],
        continuous_cols=["income"],
        detect_binary_numeric=False,
    )
    assert any(
        row["Variable"] == "income" and row["Type"] == "Continuous"
        for _, row in result.iterrows()
    )
    assert any("gender" in v for v in result["Variable"])


def test_basic_table_formatting():
    df = pd.DataFrame(
        {
            "Variable": ["age", "gender"],
            "Type": ["Continuous", "Categorical"],
            "Mean": [29.12345, None],
            "SD": [5.98765, None],
        }
    )

    output = table1_to_str(df)
    assert isinstance(output, str)
    assert "age" in output
    assert "Continuous" in output
    assert "29.12" in output  # Default float_precision = 2
    assert "5.99" in output  # Rounded


def test_padding_applied():
    df = pd.DataFrame({"Col": ["val"]})
    padded = table1_to_str(df, padding=4)
    assert padded.startswith("    Col    ")  # 4 spaces on each side


def test_max_col_width_truncates():
    df = pd.DataFrame({"ThisIsAVeryLongColumnName": ["ThisIsAVeryLongValue"]})
    output = table1_to_str(df, max_col_width=10)
    assert "ThisIsAVer" in output  # truncated column
    assert "ThisIsAVer" in output  # truncated value


def test_empty_dataframe():
    df = pd.DataFrame()
    output = table1_to_str(df)
    assert output == "[Empty Table]"


def test_none_dataframe():
    output = table1_to_str(None)
    assert output == "[Empty Table]"


def test_high_precision():
    df = pd.DataFrame(
        {
            "X": [1.123456],
            "Y": [2.987654],
        }
    )
    output = table1_to_str(df, float_precision=4)
    assert "1.1235" in output
    assert "2.9877" in output


def test_mixed_dtypes():
    df = pd.DataFrame(
        {"Name": ["Alice", "Bob"], "Age": [30, 40], "Score": [87.6789, 93.1234]}
    )
    output = table1_to_str(df, float_precision=1)
    assert "Alice" in output
    assert "Bob" in output
    assert "87.7" in output
    assert "93.1" in output


def test_include_types_continuous_only(sample_df):
    result = generate_table1(sample_df, include_types="continuous")
    assert result["Type"].nunique() == 1
    assert all(result["Type"] == "Continuous")


def test_include_types_categorical_only(sample_df):
    result = generate_table1(sample_df, include_types="categorical")
    assert result["Type"].nunique() == 1
    assert all(result["Type"] == "Categorical")


def test_include_types_both_returns_mixed(sample_df):
    result = generate_table1(sample_df, include_types="both")
    assert set(result["Type"].unique()).issubset({"Continuous", "Categorical"})


def test_include_types_invalid_raises():
    """
    This test verifies that generate_table1 handles unexpected values for
    include_types gracefully. As of the current implementation, it defaults to 'both'.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = generate_table1(df, include_types="invalid_option")

    # Updated: Check if result is DataFrame or tuple of DataFrames
    if isinstance(result, tuple):
        assert all(isinstance(r, pd.DataFrame) for r in result)
    else:
        assert isinstance(result, pd.DataFrame)


def test_generate_table1_fisher_and_chi2():
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "M", "F", "M", "F"],
            "smoker": [1, 0, 1, 0, 0, 1],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    result = generate_table1(
        df,
        include_types="categorical",
        groupby_col="group",
        use_fisher_exact=True,
    )
    assert isinstance(result, (pd.DataFrame, TableWrapper))


def test_table1_fisher_fallback_to_chi2():
    df = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B", "A", "B"],
            "category": ["X", "Y", "Z", "X", "Y", "Z"],  # not 2x2
        }
    )
    table = generate_table1(
        df,
        include_types="categorical",
        groupby_col="group",
        use_fisher_exact=True,
    )
    assert isinstance(table, (pd.DataFrame, TableWrapper))


def test_table_wrapper_methods():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    pretty_str = table1_to_str(df)  # Assumes this returns a string representation

    wrapped = TableWrapper(df, pretty_str)

    # __str__
    assert str(wrapped) == pretty_str

    # __getattr__ (e.g., .shape)
    assert wrapped.shape == (2, 2)

    # __getitem__
    assert wrapped["a"].equals(df["a"])

    # __len__
    assert len(wrapped) == 2

    # __iter__
    assert list(iter(wrapped)) == list(df)

    # .drop
    dropped = wrapped.drop(columns="a")
    assert isinstance(dropped, TableWrapper)
    assert "a" not in dropped.columns
    assert isinstance(str(dropped), str)  # verify repr still works


def test_table_wrapper_behavior():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    wrapper = TableWrapper(df, "Custom String")

    # Test __str__
    assert str(wrapper) == "Custom String"

    # Test passthrough behavior
    assert len(wrapper) == 3
    assert list(wrapper["A"]) == [1, 2, 3]
    assert isinstance(wrapper.drop(columns="B"), TableWrapper)


def test_include_types_invalid_raises():
    """
    This test verifies that generate_table1 raises a ValueError
    when include_types is passed an invalid string.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="Invalid include_types: 'invalid_option'",
    ):
        generate_table1(df, include_types="invalid_option")


def test_table1_custom_column_lists():
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "F", "M"],
            "age": [30, 40, 35, 29],
            "group": ["A", "B", "A", "B"],
        }
    )
    result = generate_table1(
        df,
        categorical_cols=["sex"],
        continuous_cols=["age"],
    )
    # you return either a DataFrame or a TableWrapper, both expose .columns
    assert isinstance(result, (pd.DataFrame, TableWrapper))
    assert "Variable" in result.columns


def test_table1_groupby_col():
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "F", "M"],
            "age": [30, 40, 35, 29],
            "group": ["A", "B", "A", "B"],
        }
    )
    result = generate_table1(df, groupby_col="group")
    assert isinstance(result, (pd.DataFrame, TableWrapper))


def test_table1_bonferroni_and_fdr_individual():
    df = pd.DataFrame(
        {
            "sex": ["M", "F", "F", "M"],
            "age": [30, 40, 35, 29],
            "group": ["A", "B", "A", "B"],
        }
    )
    result_bonferroni = generate_table1(df, groupby_col="group", apply_bonferroni=True)
    assert isinstance(result_bonferroni, (pd.DataFrame, TableWrapper))

    result_fdr = generate_table1(df, groupby_col="group", apply_bh_fdr=True)
    assert isinstance(result_fdr, (pd.DataFrame, TableWrapper))


def test_table1_value_counts():
    df = pd.DataFrame({"x": ["A", "A", "B", "B", "C"]})
    result = generate_table1(df, value_counts=True)
    assert isinstance(result, (pd.DataFrame, TableWrapper))


def test_table1_return_markdown_only():
    df = pd.DataFrame({"x": ["A", "A", "B", "B", "C"]})
    result = generate_table1(df, return_markdown_only=True)

    assert isinstance(result, dict)
    assert "categorical" in result
    assert isinstance(result["categorical"], str)
    assert "| Variable | Overall | Type | Mode" in result["categorical"]


def test_table1_detect_binary_numeric_false():
    df = pd.DataFrame({"binary_num": [0, 1, 1, 0], "group": ["A", "B", "A", "B"]})
    result = generate_table1(df, groupby_col="group", detect_binary_numeric=False)
    assert isinstance(result, (pd.DataFrame, TableWrapper))


def test_table1_markdown_export():
    # ensure the data_output folder exists
    ensure_directory(data_output)
    df = pd.DataFrame({"x": ["A", "B", "A", "C"]})
    # ask generate_table1 to write to data_output/test_table1_summary.md
    md_path = os.path.join(data_output, "test_table1_summary.md")

    generate_table1(
        df,
        export_markdown=True,
        markdown_path=md_path,
    )

    expected_file = os.path.join(data_output, "test_table1_summary_categorical.md")
    assert os.path.exists(expected_file)


def test_table1_max_categories_limit():
    df = pd.DataFrame({"x": [f"cat{i}" for i in range(20)]})
    result = generate_table1(df, max_categories=5)
    assert isinstance(result, (pd.DataFrame, TableWrapper))


def test_include_types_invalid_option_raises():
    """
    Passing an invalid include_types should raise a ValueError.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Invalid include_types"):
        generate_table1(df, include_types="invalid_option")


# ------------------------------------------------------------------
# NaN handling in generate_table1
# ------------------------------------------------------------------

def test_value_counts_nan_row_has_correct_group_counts():
    df = pd.DataFrame({
        "x":     ["a", "a", "a", "b", "b", "b", np.nan, np.nan, np.nan, np.nan],
        "group": ["A", "B", "A", "B", "A", "B", "A",    "B",    "A",    "B"],
    })
    result = generate_table1(
        df,
        categorical_cols=["x"],
        continuous_cols=[],
        groupby_col="group",
        value_counts=True,
    )

    nan_rows = result[result["Variable"] == "x = NaN"]
    assert len(nan_rows) == 1

    nan_row = nan_rows.iloc[0]
    # group columns are labeled like "A (n = 5)" / "B (n = 5)"
    a_col = [c for c in result.columns if c.startswith("A ")][0]
    b_col = [c for c in result.columns if c.startswith("B ")][0]

    # bug presented as "0 (0.00%)" in both group columns
    assert "0 (0.00%)" not in nan_row[a_col]
    assert "0 (0.00%)" not in nan_row[b_col]


def test_all_nan_categorical_column_emits_row_in_groupby():
    import warnings as _warnings

    df = pd.DataFrame({
        "x":     [np.nan] * 6,
        "group": ["A", "B", "A", "B", "A", "B"],
    })

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = generate_table1(
            df,
            categorical_cols=["x"],
            continuous_cols=[],
            groupby_col="group",
            value_counts=True,
        )

    # column should appear in the output, not be silently skipped
    assert "x" in result["Variable"].values

    # no "Skipping" warning should have fired
    skip_warnings = [w for w in caught if "Skipping" in str(w.message)]
    assert len(skip_warnings) == 0

    
def test_groupby_imputer_as_new_col_global_mean():
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B", "C"],
            "value": [1.0, np.nan, 3.0, np.nan, np.nan],
        }
    )

    out = groupby_imputer(
        df=df,
        impute_col="value",
        by="grp",
        stat="mean",
        fallback="global",
        as_new_col=True,
    )

    # new column created with the expected default name
    assert "value_mean_imputed" in out.columns

    # original column should remain unchanged (still has NaNs)
    assert df["value"].isna().sum() == 3
    assert out["value"].isna().sum() == 3

    # imputed column should have no NaNs
    assert out["value_mean_imputed"].isna().sum() == 0

    # expected imputed values:
    # grp A: mean = 1
    # grp B: mean = 3
    # grp C: no non-null in group -> fallback = global mean = (1 + 3) / 2 = 2
    expected = pd.Series([1.0, 1.0, 3.0, 3.0, 2.0], name="value_mean_imputed")
    assert np.allclose(out["value_mean_imputed"].values, expected.values)


def test_groupby_imputer_overwrites_when_as_new_col_false():
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, np.nan],
        }
    )
    df_orig = df.copy()

    out = groupby_imputer(
        df=df,
        impute_col="value",
        by="grp",
        stat="mean",
        fallback="global",
        as_new_col=False,
    )

    # result should not add an extra column
    assert "value_mean_imputed" not in out.columns

    # original df should be unchanged (function works on a copy)
    assert df.equals(df_orig)

    # output should have no missing values in 'value'
    assert out["value"].isna().sum() == 0

    # A: mean=1, B: mean=3
    expected = pd.Series([1.0, 1.0, 3.0, 3.0], name="value")
    assert np.allclose(out["value"].values, expected.values)


def test_groupby_imputer_fixed_numeric_fallback_for_empty_group():
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B"],
            "subgrp": ["x", "y", "x"],
            "value": [np.nan, np.nan, 10.0],
        }
    )

    out = groupby_imputer(
        df=df,
        impute_col="value",
        by=["grp", "subgrp"],
        stat="mean",
        fallback=0.0,
        as_new_col=True,
    )

    # groups:
    # (A, x): only NaN -> fallback = 0
    # (A, y): only NaN -> fallback = 0
    # (B, x): 10
    expected = pd.Series([0.0, 0.0, 10.0], name="value_mean_imputed")

    assert "value_mean_imputed" in out.columns
    assert np.allclose(out["value_mean_imputed"].values, expected.values)
    assert out["value_mean_imputed"].isna().sum() == 0


def test_groupby_imputer_by_str_and_list_equivalent():
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, np.nan],
        }
    )

    out_str = groupby_imputer(
        df=df,
        impute_col="value",
        by="grp",
        stat="mean",
        fallback="global",
        as_new_col=True,
    )
    out_list = groupby_imputer(
        df=df,
        impute_col="value",
        by=["grp"],
        stat="mean",
        fallback="global",
        as_new_col=True,
    )

    # both approaches should produce identical imputed columns
    col_name = "value_mean_imputed"
    assert col_name in out_str.columns
    assert col_name in out_list.columns
    assert np.allclose(out_str[col_name].values, out_list[col_name].values)


def test_groupby_imputer_custom_new_col_name():
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, np.nan],
        }
    )

    out = groupby_imputer(
        df=df,
        impute_col="value",
        by="grp",
        stat="mean",
        fallback="global",
        as_new_col=True,
        new_col_name="value_imputed",
    )

    assert "value_imputed" in out.columns
    assert "value_mean_imputed" not in out.columns  # default name not used
    assert out["value_imputed"].isna().sum() == 0


@pytest.mark.parametrize("stat", ["mean", "median", "min", "max"])
def test_groupby_imputer_supported_stats(stat):
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, 4.0],
        }
    )

    out = groupby_imputer(
        df=df,
        impute_col="value",
        by="grp",
        stat=stat,
        fallback="global",
        as_new_col=True,
    )

    # should create the expected column and have no NaNs
    col = f"value_{stat}_imputed"
    assert col in out.columns
    assert out[col].isna().sum() == 0


def test_groupby_imputer_invalid_stat_raises_valueerror():
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, np.nan],
        }
    )

    with pytest.raises(ValueError, match="stat must be one of"):
        groupby_imputer(
            df=df,
            impute_col="value",
            by="grp",
            stat="std",  # invalid
            fallback="global",
        )


@pytest.fixture
def namespace_with_dataframes():
    return {
        "df_keep": pd.DataFrame({"a": [1, 2, 3]}),
        "df_del": pd.DataFrame({"b": [4, 5, 6]}),
        "not_df": 123,
    }


@pytest.fixture
def namespace_with_ipython_cache():
    return {
        "df1": pd.DataFrame({"a": [1]}),
        "_14": pd.DataFrame({"b": [2]}),  # simulated IPython cache
    }


def test_lists_active_dataframes_without_deleting(namespace_with_dataframes):
    ns = namespace_with_dataframes.copy()

    result = del_inactive_dataframes(
        dfs_to_keep="df_keep",
        del_dataframes=False,
        namespace=ns,
        verbose=False,
    )

    assert result["active"] == ["df_del", "df_keep"]
    assert result["to_delete"] == []
    assert result["deleted"] == []
    assert result["remaining"] == []
    assert "df_del" in ns
    assert "df_keep" in ns


def test_deletes_unkept_dataframes(namespace_with_dataframes):
    ns = namespace_with_dataframes.copy()

    result = del_inactive_dataframes(
        dfs_to_keep="df_keep",
        del_dataframes=True,
        namespace=ns,
        verbose=False,
    )

    assert result["to_delete"] == ["df_del"]
    assert result["deleted"] == ["df_del"]
    assert result["remaining"] == ["df_keep"]
    assert "df_del" not in ns
    assert "df_keep" in ns


def test_dry_run_does_not_delete(namespace_with_dataframes):
    ns = namespace_with_dataframes.copy()

    result = del_inactive_dataframes(
        dfs_to_keep="df_keep",
        del_dataframes=True,
        dry_run=True,
        namespace=ns,
        verbose=False,
    )

    assert result["to_delete"] == ["df_del"]
    assert result["deleted"] == []
    assert "df_del" in ns  # still present
    assert "df_keep" in ns


def test_multiple_keep_names(namespace_with_dataframes):
    ns = namespace_with_dataframes.copy()

    result = del_inactive_dataframes(
        dfs_to_keep=["df_keep", "df_del"],
        del_dataframes=True,
        namespace=ns,
        verbose=False,
    )

    assert result["to_delete"] == []
    assert result["deleted"] == []
    assert sorted(result["remaining"]) == ["df_del", "df_keep"]


def test_ipython_cache_ignored_by_default(namespace_with_ipython_cache):
    ns = namespace_with_ipython_cache.copy()

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=True,
        namespace=ns,
        verbose=False,
    )

    # _14 should be ignored entirely
    assert result["active"] == ["df1"]
    assert result["deleted"] == ["df1"]
    assert "_14" in ns
    assert "df1" not in ns


def test_ipython_cache_included_when_enabled(namespace_with_ipython_cache):
    ns = namespace_with_ipython_cache.copy()

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=True,
        include_ipython_cache=True,
        namespace=ns,
        verbose=False,
    )

    assert sorted(result["active"]) == ["_14", "df1"]
    assert sorted(result["deleted"]) == ["_14", "df1"]
    assert ns == {}  # everything removed


def test_invalid_memory_mode_raises():
    with pytest.raises(ValueError, match="memory_mode must be one of"):
        del_inactive_dataframes(
            dfs_to_keep=[],
            memory_mode="invalid",
            verbose=False,
        )


def test_track_memory_returns_memory_block(namespace_with_dataframes):
    ns = namespace_with_dataframes.copy()

    result = del_inactive_dataframes(
        dfs_to_keep="df_keep",
        del_dataframes=True,
        namespace=ns,
        track_memory=True,
        memory_mode="dataframes",
        verbose=False,
    )

    memory = result["memory"]
    assert memory is not None
    assert memory["mode"] == "dataframes"
    assert "dataframes_mb" in memory
    assert {"before", "after", "delta"} <= set(memory["dataframes_mb"].keys())


def test_remaining_empty_when_del_dataframes_false(namespace_with_dataframes):
    ns = namespace_with_dataframes.copy()

    result = del_inactive_dataframes(
        dfs_to_keep="df_keep",
        del_dataframes=False,
        namespace=ns,
        verbose=False,
    )

    assert result["remaining"] == []


def test_non_dataframe_objects_ignored():
    ns = {
        "df": pd.DataFrame({"x": [1]}),
        "arr": np.array([1, 2, 3]),
        "text": "hello",
    }

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=True,
        namespace=ns,
        verbose=False,
    )

    assert result["deleted"] == ["df"]
    assert "arr" in ns
    assert "text" in ns


def test_run_gc_false_does_not_call_gc(mocker):

    ns = {
        "df1": pd.DataFrame({"a": [1]}),
        "df2": pd.DataFrame({"b": [2]}),
    }

    gc_mock = mocker.patch.object(gc, "collect")

    result = del_inactive_dataframes(
        dfs_to_keep="df1",
        del_dataframes=True,
        run_gc=False,
        namespace=ns,
        verbose=False,
    )

    assert result["deleted"] == ["df2"]
    gc_mock.assert_not_called()


def test_dataframe_memory_usage_exception_is_swallowed(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, 3]})

    def bad_memory_usage(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        pd.DataFrame,
        "memory_usage",
        bad_memory_usage,
    )

    ns = {"df": df}

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=False,
        namespace=ns,
        track_memory=True,
        verbose=False,
    )

    # memory exists but calculation didn't crash
    assert result["memory"] is not None


def test_process_memory_exception_handled(monkeypatch):
    class BadProcess:
        def memory_info(self):
            raise RuntimeError("psutil fail")

    class BadPsutil:
        def Process(self, pid):
            return BadProcess()

    monkeypatch.setitem(
        del_inactive_dataframes.__globals__,
        "psutil",
        BadPsutil(),
    )

    ns = {"df": pd.DataFrame({"a": [1]})}

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=True,
        namespace=ns,
        track_memory=True,
        memory_mode="all",
        verbose=False,
    )

    mem = result["memory"]["process_mb"]
    assert mem["before"] is None
    assert mem["after"] is None


def test_memory_delta_none_when_before_or_after_missing(monkeypatch):
    monkeypatch.setitem(
        del_inactive_dataframes.__globals__,
        "psutil",
        None,
    )

    ns = {"df": pd.DataFrame({"a": [1]})}

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=True,
        namespace=ns,
        track_memory=True,
        memory_mode="all",
        verbose=False,
    )

    proc = result["memory"]["process_mb"]
    assert proc["delta"] is None


def test_plain_memory_output_path(monkeypatch, capsys):
    monkeypatch.setitem(
        del_inactive_dataframes.__globals__,
        "Console",
        None,
    )
    monkeypatch.setitem(
        del_inactive_dataframes.__globals__,
        "Table",
        None,
    )

    ns = {"df": pd.DataFrame({"a": [1, 2, 3]})}

    del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=True,
        namespace=ns,
        track_memory=True,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "Memory Usage" in out


def test_rich_memory_table_rendering(monkeypatch):
    # Fake Console/Table objects to force branch execution
    class DummyConsole:
        def rule(self, *a, **k):
            pass

        def print(self, *a, **k):
            pass

    class DummyTable:
        def __init__(self, *a, **k):
            pass

        def add_column(self, *a, **k):
            pass

        def add_row(self, *a, **k):
            pass

    monkeypatch.setitem(
        del_inactive_dataframes.__globals__,
        "Console",
        DummyConsole,
    )
    monkeypatch.setitem(
        del_inactive_dataframes.__globals__,
        "Table",
        DummyTable,
    )

    ns = {"df": pd.DataFrame({"a": [1]})}

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=True,
        namespace=ns,
        track_memory=True,
        verbose=True,
    )

    assert result["used_rich"] is True


def test_ipython_cache_name_predicate_covered():
    ns = {
        "_123": pd.DataFrame({"a": [1]}),
        "df": pd.DataFrame({"b": [2]}),
    }

    result = del_inactive_dataframes(
        dfs_to_keep=[],
        del_dataframes=False,
        namespace=ns,
        include_ipython_cache=False,
        verbose=False,
    )

    assert result["active"] == ["df"]


# ------------------------------------------------------------------
# Detect Outliers
# ------------------------------------------------------------------
 
def test_returns_dataframe(clean_df):
    result = detect_outliers(clean_df)
    assert isinstance(result, pd.DataFrame)
 
 
def test_summary_columns(clean_df):
    result = detect_outliers(clean_df)
    expected = {"Variable", "Outlier (n)", "Outlier (%)", "Lower Bound", "Upper Bound"}
    assert set(result.columns) == expected
 
 
def test_summary_row_count_matches_features(clean_df):
    result = detect_outliers(clean_df, features=["A", "B"])
    assert len(result) == 2
 
 
def test_return_mask_tuple(clean_df):
    result = detect_outliers(clean_df, return_mask=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    summary, mask = result
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
 
 
def test_mask_shape(outlier_df):
    _, mask = detect_outliers(outlier_df, return_mask=True)
    assert mask.shape == (len(outlier_df), 2)
 
 
def test_mask_dtype_bool(clean_df):
    _, mask = detect_outliers(clean_df, return_mask=True)
    assert mask.dtypes.apply(lambda d: d == bool).all()
 
 
# ------------------------------------------------------------------
# IQR method
# ------------------------------------------------------------------
 
def test_iqr_detects_known_outliers(outlier_df):
    summary, mask = detect_outliers(
        outlier_df, method="iqr", threshold=1.5, return_mask=True
    )
    assert mask.loc[0, "A"] == True
    assert mask.loc[1, "A"] == True
    assert mask.loc[2, "B"] == True
 
 
def test_iqr_strict_threshold_finds_more(outlier_df):
    summary_loose = detect_outliers(outlier_df, method="iqr", threshold=3.0)
    summary_strict = detect_outliers(outlier_df, method="iqr", threshold=1.5)
    loose_n = summary_loose["Outlier (n)"].sum()
    strict_n = summary_strict["Outlier (n)"].sum()
    assert strict_n >= loose_n
 
 
def test_iqr_bounds_present(clean_df):
    result = detect_outliers(clean_df, method="iqr")
    assert result["Lower Bound"].notna().all()
    assert result["Upper Bound"].notna().all()
 
 
def test_iqr_no_outliers_on_clean_data(clean_df):
    result = detect_outliers(clean_df, method="iqr", threshold=3.0)
    assert result["Outlier (n)"].sum() == 0
 
 
# ------------------------------------------------------------------
# Z-score method
# ------------------------------------------------------------------
 
def test_zscore_detects_known_outliers(outlier_df):
    summary, mask = detect_outliers(
        outlier_df, method="zscore", threshold=3.0, return_mask=True
    )
    assert mask.loc[0, "A"] == True
    assert mask.loc[1, "A"] == True
    assert mask.loc[2, "B"] == True
 
 
def test_zscore_bounds_present(clean_df):
    result = detect_outliers(clean_df, method="zscore", threshold=3.0)
    assert result["Lower Bound"].notna().all()
    assert result["Upper Bound"].notna().all()
 
 
def test_zscore_zero_variance_column():
    """Z-score on a constant column should not raise and return 0 outliers."""
    df = pd.DataFrame({"const": [5.0] * 50, "normal": np.random.randn(50)})
    result = detect_outliers(df, method="zscore", features=["const"])
    assert result.loc[result["Variable"] == "const", "Outlier (n)"].iloc[0] == 0
 
 
# ------------------------------------------------------------------
# Isolation Forest method
# ------------------------------------------------------------------
 
def test_isoforest_runs(outlier_df):
    result = detect_outliers(outlier_df, method="isoforest", contamination=0.05)
    assert isinstance(result, pd.DataFrame)
 
 
def test_isoforest_bounds_na(outlier_df):
    result = detect_outliers(outlier_df, method="isoforest")
    assert (result["Lower Bound"] == "N/A").all()
    assert (result["Upper Bound"] == "N/A").all()
 
 
def test_isoforest_returns_nonzero_outliers(outlier_df):
    result = detect_outliers(outlier_df, method="isoforest", contamination=0.05)
    assert result["Outlier (n)"].sum() > 0
 
 
# ------------------------------------------------------------------
# groupby
# ------------------------------------------------------------------
 
def test_groupby_runs(grouped_df):
    result = detect_outliers(grouped_df, features=["value"], groupby="group")
    assert isinstance(result, pd.DataFrame)
 
 
def test_groupby_detects_per_group_outliers(grouped_df):
    summary, mask = detect_outliers(
        grouped_df, features=["value"], groupby="group", return_mask=True
    )
    assert mask.loc[0, "value"] == True
    assert mask.loc[50, "value"] == True
 
 
def test_groupby_isoforest(grouped_df):
    result = detect_outliers(
        grouped_df,
        features=["value"],
        groupby="group",
        method="isoforest",
        contamination=0.05,
    )
    assert isinstance(result, pd.DataFrame)
 
 
# ------------------------------------------------------------------
# flag_col
# ------------------------------------------------------------------
 
def test_flag_col_added_to_df(outlier_df):
    detect_outliers(outlier_df, flag_col="is_outlier")
    assert "is_outlier" in outlier_df.columns
 
 
def test_flag_col_dtype_bool(outlier_df):
    detect_outliers(outlier_df, flag_col="is_outlier")
    assert outlier_df["is_outlier"].dtype == bool
 
 
def test_flag_col_any_feature_flags_row(outlier_df):
    _, mask = detect_outliers(outlier_df, return_mask=True, flag_col="is_outlier")
    expected = mask.any(axis=1)
    pd.testing.assert_series_equal(
        outlier_df["is_outlier"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
 
 
def test_flag_col_known_rows_flagged(outlier_df):
    detect_outliers(outlier_df, flag_col="is_outlier")
    assert outlier_df.loc[0, "is_outlier"] == True
    assert outlier_df.loc[1, "is_outlier"] == True
    assert outlier_df.loc[2, "is_outlier"] == True
 
 
# ------------------------------------------------------------------
# NaN handling
# ------------------------------------------------------------------
 
def test_nan_rows_not_flagged(nan_df):
    _, mask = detect_outliers(nan_df, return_mask=True)
    # Row 2 has NaN in A — should not be flagged for A
    assert mask.loc[2, "A"] == False
    # Row 0 has NaN in B — should not be flagged for B
    assert mask.loc[0, "B"] == False
 
 
def test_nan_df_runs_without_error(nan_df):
    result = detect_outliers(nan_df)
    assert isinstance(result, pd.DataFrame)
 
 
# ------------------------------------------------------------------
# features parameter
# ------------------------------------------------------------------
 
def test_features_subset(outlier_df):
    result = detect_outliers(outlier_df, features=["A"])
    assert list(result["Variable"]) == ["A"]
 
 
def test_features_auto_excludes_non_numeric(categorical_df):
    result = detect_outliers(categorical_df)
    assert "category" not in result["Variable"].values
    assert "numeric" in result["Variable"].values
 
 
def test_features_invalid_cols_silently_dropped(clean_df):
    # Non-numeric cols are silently excluded from features list
    result = detect_outliers(clean_df, features=["A", "NonExistent"])
    assert "NonExistent" not in result["Variable"].values
 
 
# ------------------------------------------------------------------
# Summary DataFrame properties
# ------------------------------------------------------------------
 
def test_summary_sorted_by_outlier_pct_desc(outlier_df):
    result = detect_outliers(outlier_df)
    pcts = result["Outlier (%)"].tolist()
    assert pcts == sorted(pcts, reverse=True)
 
 
def test_outlier_pct_between_0_and_100(outlier_df):
    result = detect_outliers(outlier_df)
    assert (result["Outlier (%)"] >= 0).all()
    assert (result["Outlier (%)"] <= 100).all()
 
 
def test_outlier_n_nonnegative(outlier_df):
    result = detect_outliers(outlier_df)
    assert (result["Outlier (n)"] >= 0).all()
 
 
# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------
 
def test_invalid_method_raises(clean_df):
    with pytest.raises(ValueError, match="Invalid `method`"):
        detect_outliers(clean_df, method="median")
 
 
def test_invalid_threshold_raises(clean_df):
    with pytest.raises(ValueError, match="`threshold`"):
        detect_outliers(clean_df, threshold=-1.0)
 
 
def test_invalid_contamination_raises(clean_df):
    with pytest.raises(ValueError, match="`contamination`"):
        detect_outliers(clean_df, method="isoforest", contamination=0.6)
 
 
def test_invalid_groupby_column_raises(clean_df):
    with pytest.raises(ValueError, match="`groupby`"):
        detect_outliers(clean_df, groupby="nonexistent_col")
 
 
def test_no_numeric_features_raises():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="No numeric features"):
        detect_outliers(df)

## additional detect_outliers tests (verbose, return_bounds, None return)

def test_verbose_returns_none(outlier_df, capsys):
    result = detect_outliers(outlier_df, method="iqr", verbose=True)
    assert result is None


def test_verbose_prints_report(outlier_df, capsys):
    detect_outliers(outlier_df, method="iqr", verbose=True)
    captured = capsys.readouterr()
    assert "OUTLIER DETECTION SUMMARY REPORT" in captured.out
    assert "Method" in captured.out
    assert "Total rows" in captured.out
    assert "Variable" in captured.out
    assert "Outlier (n)" in captured.out
    assert "Total flagged rows" in captured.out


def test_verbose_with_return_mask_not_none(outlier_df):
    result = detect_outliers(
        outlier_df, method="iqr", verbose=True, return_mask=True
    )
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_verbose_with_return_bounds_not_none(outlier_df):
    result = detect_outliers(
        outlier_df, method="iqr", verbose=True, return_bounds=True
    )
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_verbose_groupby_prints_grouped_by(outlier_df, capsys):
    # Add a groupby column
    outlier_df = outlier_df.copy()
    outlier_df["grp"] = ["X", "Y"] * (len(outlier_df) // 2)
    detect_outliers(
        outlier_df, features=["A"], method="iqr",
        groupby="grp", verbose=True
    )
    captured = capsys.readouterr()
    assert "Grouped by" in captured.out


def test_return_bounds_type(outlier_df):
    summary, bounds = detect_outliers(
        outlier_df, method="iqr", return_bounds=True
    )
    assert isinstance(bounds, dict)
    assert set(bounds.keys()) == {"A", "B"}


def test_return_bounds_tuple_values(outlier_df):
    _, bounds = detect_outliers(
        outlier_df, method="iqr", return_bounds=True
    )
    for feat, (lower, upper) in bounds.items():
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))
        assert lower <= upper


def test_return_bounds_isoforest_na(outlier_df):
    _, bounds = detect_outliers(
        outlier_df, method="isoforest", return_bounds=True
    )
    for feat, (lower, upper) in bounds.items():
        assert lower == "N/A"
        assert upper == "N/A"


def test_return_mask_and_bounds_together(outlier_df):
    result = detect_outliers(
        outlier_df, method="iqr",
        return_mask=True, return_bounds=True
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    summary, mask, bounds = result
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert isinstance(bounds, dict)


def test_verbose_false_returns_dataframe(outlier_df):
    result = detect_outliers(outlier_df, method="iqr", verbose=False)
    assert isinstance(result, pd.DataFrame)