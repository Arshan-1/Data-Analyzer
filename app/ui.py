import streamlit as st

try:
    from app.tools import (
        build_modeled_data,
        clean_dataset,
        get_datetime_columns,
        get_basic_stats,
        get_visualization_columns,
        load_dataset,
        make_bar_chart,
        make_boxplot,
        make_corr_heatmap,
        make_histogram,
        make_scatter,
        profile_data,
        suggest_visualization,
    )
except ImportError:
    from tools import (
        build_modeled_data,
        clean_dataset,
        get_datetime_columns,
        get_basic_stats,
        get_visualization_columns,
        load_dataset,
        make_bar_chart,
        make_boxplot,
        make_corr_heatmap,
        make_histogram,
        make_scatter,
        profile_data,
        suggest_visualization,
    )

st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("AI Data Analyzer")
st.caption("Upload a tabular dataset for cleaning, analysis, and visualization.")

uploaded_file = st.file_uploader(
    "Upload data file",
    type=["csv", "xlsx", "xls", "json", "parquet"],
    help="Supported formats: CSV, Excel, JSON, and Parquet.",
)

if uploaded_file:
    try:
        raw_df = load_dataset(uploaded_file)
    except Exception as error:
        st.error(f"Could not read file: {error}")
        st.stop()

    if raw_df.empty:
        st.warning("The uploaded file has no rows.")
        st.stop()

    cleaned_df, clean_report = clean_dataset(raw_df)
    profile = profile_data(cleaned_df)
    modeled_df = cleaned_df.copy()

    tab_overview, tab_cleaning, tab_analysis, tab_modeling, tab_visual = st.tabs(
        ["Overview", "Cleaning", "Analysis", "Modeling", "Visualization"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", profile["shape"][0])
        c2.metric("Columns", profile["shape"][1])
        c3.metric("Missing Values", sum(profile["missing_values"].values()))
        c4.metric("Duplicate Rows", profile["duplicate_rows"])

        st.subheader("Preview")
        st.dataframe(cleaned_df.head(50), use_container_width=True)

        with st.expander("Column Types"):
            st.json(profile["dtypes"])

    with tab_cleaning:
        st.subheader("Cleaning Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Before", clean_report["rows_before"])
        c2.metric("Rows After", clean_report["rows_after"])
        c3.metric("Duplicates Removed", clean_report["duplicates_removed"])
        c4.metric("Missing After", clean_report["missing_after"])
        st.caption("Numeric NaNs are median-filled; non-numeric NaNs are mode-filled.")

    with tab_analysis:
        st.subheader("Descriptive Statistics")
        stats_df = get_basic_stats(cleaned_df)
        st.dataframe(stats_df, use_container_width=True)

    with tab_modeling:
        st.subheader("Data Modeling")
        st.caption("Create transformed data that will be used in final visualization.")

        model_type = st.selectbox(
            "Modeling type",
            ["None", "Aggregate by Category", "Time Trend"],
        )
        agg_func = st.selectbox("Aggregation method", ["sum", "mean", "median", "min", "max"])

        numeric_cols_clean, categorical_cols_clean = get_visualization_columns(cleaned_df)
        datetime_cols_clean = get_datetime_columns(cleaned_df)

        try:
            if model_type == "Aggregate by Category":
                if not categorical_cols_clean or not numeric_cols_clean:
                    st.warning("Need at least one categorical and one numeric column for this model.")
                else:
                    group_col = st.selectbox("Category column", categorical_cols_clean)
                    value_col = st.selectbox("Value column", numeric_cols_clean)
                    top_n = st.slider("Top categories to keep", 5, 100, 20)
                    modeled_df = build_modeled_data(
                        cleaned_df,
                        model_type=model_type,
                        group_col=group_col,
                        value_col=value_col,
                        agg_func=agg_func,
                        top_n=top_n,
                    )
            elif model_type == "Time Trend":
                if not datetime_cols_clean or not numeric_cols_clean:
                    st.warning("Need at least one datetime and one numeric column for this model.")
                else:
                    date_col = st.selectbox("Date column", datetime_cols_clean)
                    value_col = st.selectbox("Value column", numeric_cols_clean, key="trend_value_col")
                    freq = st.selectbox("Time frequency", ["D", "W", "M", "Q", "Y"])
                    modeled_df = build_modeled_data(
                        cleaned_df,
                        model_type=model_type,
                        date_col=date_col,
                        value_col=value_col,
                        agg_func=agg_func,
                        freq=freq,
                    )
            else:
                modeled_df = cleaned_df.copy()
        except Exception as error:
            st.error(f"Modeling failed: {error}")
            modeled_df = cleaned_df.copy()

        st.write("Modeled data preview")
        st.dataframe(modeled_df.head(50), use_container_width=True)

    with tab_visual:
        numeric_cols, categorical_cols = get_visualization_columns(modeled_df)
        all_cols = modeled_df.columns.tolist()
        suggestion = suggest_visualization(modeled_df)

        st.subheader("Visualization")
        st.caption(f"Auto suggestion: {suggestion['reason']}")

        chart_options = [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Correlation Heatmap",
        ]

        if "viz_chart" not in st.session_state:
            st.session_state["viz_chart"] = suggestion["chart"]
        if "viz_x" not in st.session_state:
            st.session_state["viz_x"] = suggestion["x"]
        if "viz_y" not in st.session_state:
            st.session_state["viz_y"] = suggestion["y"]

        if st.button("Use Auto Suggestion"):
            st.session_state["viz_chart"] = suggestion["chart"]
            st.session_state["viz_x"] = suggestion["x"]
            st.session_state["viz_y"] = suggestion["y"]

        viz_type = st.selectbox(
            "Choose chart",
            chart_options,
            key="viz_chart",
        )
        x_col = st.selectbox("X-axis column", [""] + all_cols, key="viz_x")
        y_col = st.selectbox("Y-axis column", [""] + all_cols, key="viz_y")

        if viz_type == "Histogram":
            if x_col and x_col in numeric_cols:
                st.plotly_chart(make_histogram(modeled_df, x_col), use_container_width=True)
            else:
                st.info("Select a numeric X-axis column.")

        elif viz_type == "Box Plot":
            if y_col and y_col in numeric_cols:
                st.plotly_chart(make_boxplot(modeled_df, y_col), use_container_width=True)
            else:
                st.info("Select a numeric Y-axis column.")

        elif viz_type == "Bar Chart":
            if x_col and y_col:
                st.bar_chart(modeled_df, x=x_col, y=y_col)
            elif x_col and x_col in categorical_cols:
                st.plotly_chart(make_bar_chart(modeled_df, x_col), use_container_width=True)
            else:
                st.info("Select X-axis (category). Y-axis is optional.")

        elif viz_type == "Line Chart":
            if x_col and y_col:
                st.line_chart(modeled_df, x=x_col, y=y_col)
            else:
                st.info("Select both X-axis and Y-axis columns.")

        elif viz_type == "Scatter Plot":
            if x_col and y_col:
                st.plotly_chart(make_scatter(modeled_df, x_col, y_col), use_container_width=True)
            else:
                st.info("Select both X-axis and Y-axis columns.")

        elif viz_type == "Correlation Heatmap":
            heatmap = make_corr_heatmap(modeled_df)
            if heatmap is None:
                st.info("No numeric columns available for correlation heatmap.")
            else:
                st.plotly_chart(heatmap, use_container_width=True)