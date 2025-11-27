"""
Time Series Forecasting Pipeline
Supports multiple forecasting methods with AI-driven decision making
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

from src.timeseries_agent import TimeSeriesAgent
from src.ai_summarizer import summarize_timeseries_results


def timeseries_model_pipeline(dataframe, API_KEY, GPT_MODEL):
    """Main pipeline for time series forecasting"""

    st.header("⏰ Time Series Forecasting")
    st.write("AI-powered time series analysis and forecasting")

    # Initialize time series agent
    if 'ts_agent' not in st.session_state:
        st.session_state.ts_agent = TimeSeriesAgent(API_KEY, GPT_MODEL)

    # Step 1: Data Inspection and Column Selection
    st.subheader("📊 Step 1: Data Inspection")

    with st.expander("📋 Dataset Preview", expanded=True):
        st.write(f"**Dataset Shape:** {dataframe.shape[0]} rows × {dataframe.shape[1]} columns")
        st.dataframe(dataframe.head(10))
        st.write("**Column Data Types:**")
        st.write(dataframe.dtypes)

    # Step 2: AI-Assisted Column Selection
    st.subheader("🤖 Step 2: AI Column Detection")

    if 'ts_columns_identified' not in st.session_state:
        with st.spinner("AI is analyzing your data to identify time series columns..."):
            column_analysis = st.session_state.ts_agent.identify_time_columns(dataframe)
            st.session_state.ts_date_column = column_analysis.get('date_column')
            st.session_state.ts_value_column = column_analysis.get('value_column')
            st.session_state.ts_columns_identified = True

    # Allow manual override
    col1, col2 = st.columns(2)
    with col1:
        date_column = st.selectbox(
            "Date/Time Column",
            dataframe.columns.tolist(),
            index=dataframe.columns.tolist().index(st.session_state.ts_date_column) if st.session_state.ts_date_column in dataframe.columns else 0
        )
    with col2:
        value_column = st.selectbox(
            "Value Column (to forecast)",
            dataframe.columns.tolist(),
            index=dataframe.columns.tolist().index(st.session_state.ts_value_column) if st.session_state.ts_value_column in dataframe.columns else 0
        )

    if st.button("🔍 Analyze Time Series", type="primary"):
        st.session_state.ts_analysis_started = True
        st.session_state.ts_date_col = date_column
        st.session_state.ts_value_col = value_column

    if st.session_state.get('ts_analysis_started', False):
        # Step 3: Data Preprocessing
        st.subheader("🔧 Step 3: Data Preprocessing")

        with st.spinner("AI is preprocessing your time series data..."):
            ts_data = preprocess_timeseries(
                dataframe,
                st.session_state.ts_date_col,
                st.session_state.ts_value_col,
                API_KEY,
                GPT_MODEL
            )

            if ts_data is None:
                st.error("Failed to preprocess time series data. Please check your date and value columns.")
                return

            st.session_state.ts_data = ts_data

        st.success(f"✅ Preprocessed {len(ts_data)} time points")

        # Step 4: Exploratory Analysis
        st.subheader("📈 Step 4: Time Series Visualization")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts_data.index, ts_data.values, linewidth=2)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(st.session_state.ts_value_col, fontsize=12)
        ax.set_title('Time Series Data', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Step 5: AI Analysis and Method Selection
        st.subheader("🤖 Step 5: AI Analysis")

        if 'ts_analysis_complete' not in st.session_state:
            with st.spinner("AI is analyzing patterns and selecting forecasting methods..."):
                analysis = st.session_state.ts_agent.analyze_timeseries(ts_data)
                st.session_state.ts_analysis = analysis
                st.session_state.ts_analysis_complete = True

        with st.expander("📊 AI Analysis Results", expanded=True):
            st.write("**Detected Patterns:**")
            for pattern in st.session_state.ts_analysis.get('patterns', []):
                st.write(f"- {pattern}")

            st.write("\n**Recommended Methods:**")
            for method in st.session_state.ts_analysis.get('recommended_methods', []):
                st.write(f"- {method}")

            st.write(f"\n**Forecast Horizon:** {st.session_state.ts_analysis.get('forecast_horizon', 30)} periods")

        # Step 6: Model Training and Forecasting
        st.subheader("🎯 Step 6: Model Training & Forecasting")

        forecast_periods = st.slider(
            "Forecast Horizon (periods ahead)",
            min_value=5,
            max_value=100,
            value=st.session_state.ts_analysis.get('forecast_horizon', 30)
        )

        if st.button("🚀 Start Forecasting", type="primary"):
            st.session_state.ts_forecast_started = True
            st.session_state.ts_forecast_periods = forecast_periods

        if st.session_state.get('ts_forecast_started', False):
            train_and_display_forecasts(
                st.session_state.ts_data,
                st.session_state.ts_forecast_periods,
                st.session_state.ts_analysis,
                st.session_state.ts_value_col,
                API_KEY,
                GPT_MODEL
            )


def preprocess_timeseries(dataframe, date_column, value_column, API_KEY, GPT_MODEL):
    """Preprocess time series data with AI assistance"""

    try:
        # Create a copy
        df = dataframe[[date_column, value_column]].copy()

        # Convert date column
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Remove rows with invalid dates
        df = df.dropna(subset=[date_column])

        # Convert value column to numeric
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

        # Handle missing values
        if df[value_column].isna().sum() > 0:
            # AI decides best imputation method
            ts_agent = TimeSeriesAgent(API_KEY, GPT_MODEL)
            missing_pct = (df[value_column].isna().sum() / len(df)) * 100

            if missing_pct < 5:
                # Forward fill for small gaps
                df[value_column] = df[value_column].fillna(method='ffill').fillna(method='bfill')
            else:
                # Interpolate for larger gaps
                df[value_column] = df[value_column].interpolate(method='time')

        # Set date as index
        df = df.set_index(date_column).sort_index()

        # Create time series
        ts_data = df[value_column]

        # Remove duplicates (keep last)
        ts_data = ts_data[~ts_data.index.duplicated(keep='last')]

        return ts_data

    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None


def train_and_display_forecasts(ts_data, forecast_periods, analysis, value_col_name, API_KEY, GPT_MODEL):
    """Train multiple models and display forecasts"""

    # Split data into train/test
    test_size = min(int(len(ts_data) * 0.2), forecast_periods)
    train_data = ts_data[:-test_size]
    test_data = ts_data[-test_size:]

    st.write(f"**Training on:** {len(train_data)} points | **Testing on:** {len(test_data)} points")

    # Store results
    results = {}
    forecasts = {}

    # Model 1: ARIMA
    with st.spinner("Training ARIMA model..."):
        arima_result = train_arima(train_data, test_data, forecast_periods)
        if arima_result:
            results['ARIMA'] = arima_result['metrics']
            forecasts['ARIMA'] = arima_result['forecast']

    # Model 2: Exponential Smoothing
    with st.spinner("Training Exponential Smoothing model..."):
        es_result = train_exponential_smoothing(train_data, test_data, forecast_periods)
        if es_result:
            results['Exponential Smoothing'] = es_result['metrics']
            forecasts['Exponential Smoothing'] = es_result['forecast']

    # Model 3: Prophet-like Trend + Seasonality
    with st.spinner("Training Trend+Seasonality model..."):
        trend_result = train_trend_model(train_data, test_data, forecast_periods)
        if trend_result:
            results['Trend Model'] = trend_result['metrics']
            forecasts['Trend Model'] = trend_result['forecast']

    # Display Results
    display_forecast_results(
        ts_data,
        train_data,
        test_data,
        forecasts,
        results,
        value_col_name,
        API_KEY,
        GPT_MODEL
    )


def train_arima(train_data, test_data, forecast_periods):
    """Train ARIMA model"""
    try:
        # Auto-select order (simplified)
        model = ARIMA(train_data, order=(1, 1, 1))
        fitted_model = model.fit()

        # Forecast
        forecast = fitted_model.forecast(steps=len(test_data) + forecast_periods)
        test_forecast = forecast[:len(test_data)]
        future_forecast = forecast[len(test_data):]

        # Calculate metrics
        mse = mean_squared_error(test_data, test_forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, test_forecast)

        return {
            'model': fitted_model,
            'forecast': forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MSE': mse}
        }
    except Exception as e:
        st.warning(f"ARIMA training failed: {str(e)}")
        return None


def train_exponential_smoothing(train_data, test_data, forecast_periods):
    """Train Exponential Smoothing model"""
    try:
        # Determine if seasonal
        seasonal_periods = None
        if len(train_data) >= 24:
            seasonal_periods = 12  # Assume monthly seasonality

        if seasonal_periods and len(train_data) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                train_data,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            )
        else:
            model = ExponentialSmoothing(train_data, trend='add')

        fitted_model = model.fit()

        # Forecast
        forecast = fitted_model.forecast(steps=len(test_data) + forecast_periods)
        test_forecast = forecast[:len(test_data)]

        # Calculate metrics
        mse = mean_squared_error(test_data, test_forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, test_forecast)

        return {
            'model': fitted_model,
            'forecast': forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MSE': mse}
        }
    except Exception as e:
        st.warning(f"Exponential Smoothing training failed: {str(e)}")
        return None


def train_trend_model(train_data, test_data, forecast_periods):
    """Train trend-based model with ML"""
    try:
        # Create features
        train_df = pd.DataFrame({
            'value': train_data.values,
            'time_idx': range(len(train_data))
        })

        # Add lag features
        for lag in [1, 7, 30]:
            if len(train_data) > lag:
                train_df[f'lag_{lag}'] = train_data.shift(lag).values

        train_df = train_df.dropna()

        X_train = train_df.drop('value', axis=1)
        y_train = train_df['value']

        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Forecast on test
        test_forecasts = []
        for i in range(len(test_data) + forecast_periods):
            time_idx = len(train_data) + i
            features = {'time_idx': time_idx}

            # Add lag features
            if i == 0:
                features['lag_1'] = train_data.iloc[-1]
                features['lag_7'] = train_data.iloc[-7] if len(train_data) >= 7 else train_data.iloc[0]
                features['lag_30'] = train_data.iloc[-30] if len(train_data) >= 30 else train_data.iloc[0]
            else:
                features['lag_1'] = test_forecasts[-1] if i > 0 else train_data.iloc[-1]
                features['lag_7'] = test_forecasts[-7] if i >= 7 else train_data.iloc[-7] if len(train_data) >= 7 else train_data.iloc[0]
                features['lag_30'] = test_forecasts[-30] if i >= 30 else train_data.iloc[-30] if len(train_data) >= 30 else train_data.iloc[0]

            X_pred = pd.DataFrame([features])
            pred = model.predict(X_pred)[0]
            test_forecasts.append(pred)

        forecast = pd.Series(test_forecasts)
        test_forecast = forecast[:len(test_data)]

        # Calculate metrics
        mse = mean_squared_error(test_data, test_forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, test_forecast)

        return {
            'model': model,
            'forecast': forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MSE': mse}
        }
    except Exception as e:
        st.warning(f"Trend model training failed: {str(e)}")
        return None


def display_forecast_results(ts_data, train_data, test_data, forecasts, results, value_col_name, API_KEY, GPT_MODEL):
    """Display forecasting results and visualizations"""

    st.subheader("📊 Forecasting Results")

    # Display metrics comparison
    st.write("**Model Performance Comparison (on test set):**")

    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df.round(4)
    st.dataframe(metrics_df)

    # Highlight best model
    best_model = metrics_df['RMSE'].idxmin()
    st.success(f"🏆 **Best Model:** {best_model} (Lowest RMSE)")

    # Visualize forecasts
    st.subheader("📈 Forecast Visualizations")

    # Create tabs for different models
    model_tabs = st.tabs(list(forecasts.keys()))

    for idx, (model_name, forecast) in enumerate(forecasts.items()):
        with model_tabs[idx]:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Plot historical data
            ax.plot(train_data.index, train_data.values, label='Training Data', linewidth=2, color='blue')
            ax.plot(test_data.index, test_data.values, label='Test Data', linewidth=2, color='green')

            # Plot forecast
            forecast_index = pd.date_range(
                start=test_data.index[0],
                periods=len(forecast),
                freq=pd.infer_freq(train_data.index) or 'D'
            )
            ax.plot(forecast_index, forecast.values, label='Forecast', linewidth=2, color='red', linestyle='--')

            # Confidence interval (simplified)
            std_error = results[model_name]['RMSE']
            ax.fill_between(
                forecast_index,
                forecast.values - 1.96 * std_error,
                forecast.values + 1.96 * std_error,
                alpha=0.2,
                color='red',
                label='95% Confidence Interval'
            )

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(value_col_name, fontsize=12)
            ax.set_title(f'{model_name} Forecast', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close()

            # Metrics for this model
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{results[model_name]['RMSE']:.4f}")
            with col2:
                st.metric("MAE", f"{results[model_name]['MAE']:.4f}")
            with col3:
                st.metric("MSE", f"{results[model_name]['MSE']:.4f}")

    # Download forecasts
    st.divider()
    st.subheader("📥 Download Forecasts")

    download_col1, download_col2, download_col3 = st.columns(3)

    for idx, (model_name, forecast) in enumerate(forecasts.items()):
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(
                start=test_data.index[0],
                periods=len(forecast),
                freq=pd.infer_freq(train_data.index) or 'D'
            ),
            'Forecast': forecast.values
        })

        csv_data = forecast_df.to_csv(index=False).encode('utf-8')

        with [download_col1, download_col2, download_col3][idx % 3]:
            st.download_button(
                label=f"Download {model_name}",
                data=csv_data,
                file_name=f'{model_name.lower().replace(" ", "_")}_forecast.csv',
                mime='text/csv',
                key=f'download_{model_name}'
            )

    # AI Summary
    st.divider()
    st.subheader("🤖 AI Analysis Summary")

    with st.spinner("AI is analyzing the forecasting results..."):
        user_query = st.session_state.get('user_query', None)

        summary = summarize_timeseries_results(
            best_model=best_model,
            model_metrics=results,
            forecast_horizon=len(test_data),
            api_key=API_KEY,
            model_type=GPT_MODEL,
            user_query=user_query
        )

        st.markdown(summary)
