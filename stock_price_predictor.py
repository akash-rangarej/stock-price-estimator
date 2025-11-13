import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# App title and description
st.title("📈 Stock Price Prediction App")
st.markdown("""
This app predicts stock prices based on technical indicators and market data.
Enter the required parameters below and click **Predict** to get the estimated price.
""")

# Sidebar for additional information
st.sidebar.header("About")
st.sidebar.info(
    "This machine learning model uses historical stock data and technical indicators "
    "to predict future stock prices. The model was trained on various market features."
)

# Main input section
st.header("📊 Input Parameters")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Data")
    open_price = st.number_input("Open Price", step=1.0)
    high_price = st.number_input("High Price", step=1.0)
    low_price = st.number_input("Low Price",step=1.0)
    volume = st.number_input("Volume", min_value=0, value=1000000, step=1000)

with col2:
    st.subheader("Technical Indicators")
    ma50 = st.number_input("50-Day Moving Average (MA50)", step=1.0)
    ma200 = st.number_input("200-Day Moving Average (MA200)", step=1.0)
    
    st.subheader("Date Information")
    year = st.number_input("Year", min_value=1995, max_value=2030, value=2024, step=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)

# Prediction button
if st.button("🚀 Predict Stock Price", type="primary"):
    try:
        # Create feature array
        features = [open_price, high_price, low_price, ma50, ma200, volume, year, month, day]
        
        # Create DataFrame
        columns =['Prev_Open','Prev_High','Prev_Low','MA50_prev','MA200_prev','Prev_Volume','Year','Month','Day']
        test_df = pd.DataFrame(np.array(features).reshape(1, -1), columns=columns)
        
        # Load model and make prediction
        model = joblib.load('stock_price_predictor.pkl')
        pred = model.predict(test_df)
        
        # Display results
        st.success("✅ Prediction completed successfully!")
        
        # Create a nice results section
        st.header("📋 Results")
        
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.subheader("Input Summary")
            st.write(f"**Open:** ${open_price:.2f}")
            st.write(f"**High:** ${high_price:.2f}")
            st.write(f"**Low:** ${low_price:.2f}")
            st.write(f"**Volume:** {volume:,}")
            st.write(f"**MA50:** ${ma50:.2f}")
            st.write(f"**MA200:** ${ma200:.2f}")
            st.write(f"**Date:** {year}-{month:02d}-{day:02d}")
        
        with results_col2:
            st.subheader("Prediction")
            st.metric(
                label="Estimated Stock Price",
                value=f"${pred[0]:.2f}",
                delta=None
            )
            
            # Additional visual feedback
            if pred[0] > open_price:
                st.success("📈 Bullish signal: Predicted price is higher than opening price")
            else:
                st.warning("📉 Bearish signal: Predicted price is lower than opening price")
    
    except FileNotFoundError:
        st.error("❌ Model file 'stock_price_predictor.pkl' not found. Please make sure the model file is in the same directory.")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# Additional information section
st.markdown("---")
st.header("ℹ️ How to Use")
st.markdown("""
1. **Open Price**: The stock's opening price previous day
2. **High Price**: The highest price reached during the previous trading day
3. **Low Price**: The lowest price reached during the  previous trading day
4. **MA50**: 50-day moving average - average of last 50 days' previous closing prices
5. **MA200**: 200-day moving average - average of last 200 days' previous closing prices
6. **Volume**: Number of shares traded in previous day
7. **Date**: The specific date for prediction

Fill in all the fields and click the **Predict** button to get the estimated stock price.
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Stock Price Prediction App | Made with Streamlit"
    "</div>",
    unsafe_allow_html=True
)