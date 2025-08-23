import streamlit as st
import os
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"
import pandas as pd
from Top30_Stocks import load_consolidated_data, rank_with_forecast, backtest_selector, detect_and_read_raw, find_header_row_with_column, read_with_detected_header


st.set_page_config(page_title="Top 30 Stock Selector", layout="wide")

st.title("📈 Top 30 Stock Selector")
st.markdown("Upload your stock data (CSV or Excel) and run the analysis.")

# File upload
uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = load_consolidated_data(uploaded_file)
    if df is not None and not df.empty:
        st.success(f"Loaded {len(df)} rows.")
    else:
        st.warning("No data loaded. Check the file format and contents.")
    
    # Show preview
    with st.expander("🔍 Preview Data"):
        st.write(df.head())

    # Buttons
    if st.button("🏆 Run Top 30 Ranking"):
        ranked_df = rank_stocks(df, top_n=30)
        st.subheader("Top 30 Stocks")
        st.dataframe(ranked_df)

        # Optionally save
        st.download_button(
            label="Download Results as CSV",
            data=ranked_df.to_csv(index=False),
            file_name="top30_stocks.csv",
            mime="text/csv"
        )

    if st.button("📊 Run Backtest"):
        results = backtest_strategy(df)
        st.subheader("Backtest Results")
        st.write(results)
