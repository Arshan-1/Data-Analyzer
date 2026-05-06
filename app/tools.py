import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


MISSING_TOKENS = ["", "na", "n/a", "null", "none", "nan", "missing", "-", "--"]


def load_dataset(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        excel_file = pd.ExcelFile(uploaded_file)
        best_df = None
        best_size = -1
        for sheet_name in excel_file.sheet_names:
            sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if not sheet_df.empty and sheet_df.shape[1] > 0:
                sheet_size = sheet_df.shape[0] * sheet_df.shape[1]
                if sheet_size > best_size:
                    best_size = sheet_size
                    best_df = sheet_df
        if best_df is not None:
            return best_df
        raise ValueError(
            "Excel file was loaded, but no non-empty sheet was found."
        )
    if file_name.endswith(".json"):
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        try:
            return pd.read_json(io.BytesIO(raw))
        except ValueError:
            return pd.read_json(io.BytesIO(raw), lines=True)
    if file_name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV, Excel, JSON, or Parquet.")


def _normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype("string").str.strip()
        cleaned[col] = cleaned[col].replace(MISSING_TOKENS, pd.NA)
    return cleaned


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    cleaned = _normalize_missing_values(df)
    report = {
        "rows_before": len(cleaned),
        "duplicates_removed": 0,
        "rows_after": 0,
        "missing_before": int(df.isna().sum().sum()),
        "missing_after": 0,
    }

    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = duplicate_count

    for col in cleaned.columns:
        if cleaned[col].dtype == "object" or str(cleaned[col].dtype).startswith("string"):
            numeric_try = pd.to_numeric(cleaned[col], errors="coerce")
            if numeric_try.notna().mean() > 0.8:
                cleaned[col] = numeric_try
                continue

            datetime_try = pd.to_datetime(cleaned[col], errors="coerce")
            if datetime_try.notna().mean() > 0.8:
                cleaned[col] = datetime_try

    for col in cleaned.select_dtypes(include=[np.number]).columns:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in cleaned.select_dtypes(exclude=[np.number]).columns:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    report["rows_after"] = len(cleaned)
    report["missing_after"] = int(cleaned.isna().sum().sum())
    return cleaned, report


def profile_data(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)


def detect_outliers(df: pd.DataFrame, column: str) -> int:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)]
    return len(outliers)


def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose()


def get_visualization_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def make_histogram(df: pd.DataFrame, column: str):
    return px.histogram(df, x=column, title=f"Distribution of {column}")


def make_boxplot(df: pd.DataFrame, column: str):
    return px.box(df, y=column, title=f"Box Plot of {column}")


def make_bar_chart(df: pd.DataFrame, column: str):
    counts = df[column].value_counts().head(20).reset_index()
    counts.columns = [column, "count"]
    return px.bar(counts, x=column, y="count", title=f"Top Categories in {column}")


def make_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    return px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")


def make_corr_heatmap(df: pd.DataFrame):
    corr = correlation_matrix(df)
    if corr.empty:
        return None
    return ff.create_annotated_heatmap(
        z=corr.round(2).values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="Viridis",
        showscale=True,
    )


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()


def build_modeled_data(
    df: pd.DataFrame,
    model_type: str,
    group_col: str = "",
    value_col: str = "",
    agg_func: str = "sum",
    date_col: str = "",
    freq: str = "M",
    top_n: int = 20,
) -> pd.DataFrame:
    if model_type == "None":
        return df.copy()

    if model_type == "Aggregate by Category":
        if not group_col or not value_col:
            raise ValueError("Please select both group and value columns for aggregation.")
        modeled = (
            df.groupby(group_col, dropna=False)[value_col]
            .agg(agg_func)
            .reset_index()
            .sort_values(by=value_col, ascending=False)
            .head(top_n)
        )
        return modeled

    if model_type == "Time Trend":
        if not date_col or not value_col:
            raise ValueError("Please select both date and value columns for time trend.")
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])
        if temp.empty:
            raise ValueError("No valid datetime values found in selected date column.")
        modeled = (
            temp.set_index(date_col)[value_col]
            .resample(freq)
            .agg(agg_func)
            .reset_index()
            .sort_values(by=date_col)
        )
        return modeled

    return df.copy()


def suggest_visualization(df: pd.DataFrame) -> Dict[str, str]:
    numeric_cols, categorical_cols = get_visualization_columns(df)
    datetime_cols = get_datetime_columns(df)

    if datetime_cols and numeric_cols:
        return {
            "chart": "Line Chart",
            "x": datetime_cols[0],
            "y": numeric_cols[0],
            "color": "",
            "reason": "Detected time-based and numeric columns; line chart is suggested.",
        }

    if categorical_cols and numeric_cols:
        return {
            "chart": "Bar Chart",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "color": "",
            "reason": "Detected categorical + numeric columns; bar chart is suggested.",
        }

    if len(numeric_cols) >= 2:
        return {
            "chart": "Scatter Plot",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "color": "",
            "reason": "Detected multiple numeric columns; scatter plot is suggested.",
        }

    if len(numeric_cols) == 1:
        return {
            "chart": "Histogram",
            "x": numeric_cols[0],
            "y": "",
            "color": "",
            "reason": "Detected one numeric column; histogram is suggested.",
        }

    if categorical_cols:
        return {
            "chart": "Bar Chart",
            "x": categorical_cols[0],
            "y": "",
            "color": "",
            "reason": "Detected categorical data; category frequency bar chart is suggested.",
        }

    return {
        "chart": "Histogram",
        "x": "",
        "y": "",
        "color": "",
        "reason": "Could not infer a strong chart suggestion.",
    }