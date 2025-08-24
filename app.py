# app.py
import streamlit as st
import os
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"
import pandas as pd
from select_top30 import (
    load_consolidated_data,
    rank_with_forecast,
    backtest_selector,
    selector_for_backtest,
    selector_conservative
)

st.set_page_config(page_title="Top 30 Stock Selector", layout="wide")

st.title("📈 Top 30 Stock Selector")
st.markdown("Upload your stock data (CSV or Excel) and run the analysis.")

# File upload
uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = load_consolidated_data(uploaded_file)
        if df is not None and not df.empty:
            st.success(f"✅ Loaded {len(df)} rows and {len(df['Symbol'].unique())} unique stocks.")
        else:
            st.warning("No data loaded. Check the file format and contents.")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        df = None

    if df is not None and not df.empty:
        # Show preview
        with st.expander("🔍 Preview Data"):
            st.dataframe(df.head())

        # Ranking
        if st.button("🏆 Run Top 30 Ranking"):
            try:
                ranked_df = rank_with_forecast(df, top_n=30)
                st.subheader("Top 30 Stocks")
                st.dataframe(ranked_df)

                st.download_button(
                    label="Download Results as CSV",
                    data=ranked_df.to_csv(index=False),
                    file_name="top30_stocks.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Ranking failed: {e}")

        # Ultra-conservative bucket
        if st.button("🛡️ Ultra-Conservative Safe Bucket"):
            try:
                safe_bucket = selector_conservative(df, top_n=30)
                if not safe_bucket.empty:
                    st.subheader("Ultra-Conservative Safe Picks")
                    st.dataframe(safe_bucket)
                    st.download_button(
                        label="Download Safe Bucket CSV",
                        data=safe_bucket.to_csv(index=False),
                        file_name="safe_bucket.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No stocks met ultra-conservative criteria.")
            except Exception as e:
                st.error(f"Safe bucket computation failed: {e}")

        # Backtesting
        if st.button("📊 Run Backtest"):
            try:
                summary_df, agg = backtest_selector(df, selector_fn=selector_for_backtest)
                st.subheader("Backtest Results - Per Run")
                st.dataframe(summary_df)
                st.subheader("Backtest Aggregate")
                st.write(agg)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
