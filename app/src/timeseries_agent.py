"""
Time Series Agent
AI-powered decision making for time series analysis and forecasting
"""

import pandas as pd
import numpy as np
from src.llm_service import call_openrouter, get_model_name
import json


class TimeSeriesAgent:
    """Agent that analyzes time series data and recommends optimal approaches"""

    def __init__(self, api_key, model_type=4):
        self.api_key = api_key
        self.model_type = model_type

    def identify_time_columns(self, dataframe):
        """Identify date/time column and value column using AI"""

        # Prepare data summary
        column_info = []
        for col in dataframe.columns:
            dtype = str(dataframe[col].dtype)
            sample_values = dataframe[col].dropna().head(5).tolist()
            null_count = dataframe[col].isna().sum()

            column_info.append({
                'name': col,
                'dtype': dtype,
                'sample_values': [str(v) for v in sample_values],
                'null_count': null_count,
                'total_rows': len(dataframe)
            })

        system_prompt = """You are a time series data expert. Analyze the dataset columns and identify:
1. The DATE/TIME column (contains dates, timestamps, or time indicators)
2. The VALUE column (the numeric variable to forecast)

Return ONLY a JSON object with this exact format:
{
    "date_column": "column_name_here",
    "value_column": "column_name_here",
    "reasoning": "brief explanation"
}"""

        user_message = f"""Analyze these columns and identify the date/time column and value column for time series forecasting:

{json.dumps(column_info, indent=2)}

Return JSON only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            model_name = get_model_name(self.model_type)
            response = call_openrouter(
                messages=messages,
                model_name=model_name,
                api_key=self.api_key,
                function_name="identify_time_columns"
            )

            # Parse JSON response
            result = self._extract_json(response)
            return result

        except Exception as e:
            # Fallback: simple heuristics
            return self._fallback_column_detection(dataframe)

    def analyze_timeseries(self, ts_data):
        """Analyze time series patterns and recommend forecasting methods"""

        # Calculate statistics
        stats = {
            'length': len(ts_data),
            'mean': float(ts_data.mean()),
            'std': float(ts_data.std()),
            'min': float(ts_data.min()),
            'max': float(ts_data.max()),
            'has_trend': self._detect_trend(ts_data),
            'has_seasonality': self._detect_seasonality(ts_data),
            'stationarity': self._check_stationarity(ts_data)
        }

        system_prompt = """You are a time series forecasting expert. Based on the data characteristics, recommend:
1. Detected patterns (trend, seasonality, cyclical, irregular)
2. Best forecasting methods (ARIMA, Exponential Smoothing, ML-based, etc.)
3. Recommended forecast horizon

Return ONLY a JSON object with this exact format:
{
    "patterns": ["pattern1", "pattern2"],
    "recommended_methods": ["method1", "method2", "method3"],
    "forecast_horizon": 30,
    "reasoning": "brief explanation"
}"""

        user_message = f"""Analyze this time series data:

Length: {stats['length']} data points
Mean: {stats['mean']:.2f}
Std Dev: {stats['std']:.2f}
Range: [{stats['min']:.2f}, {stats['max']:.2f}]
Trend detected: {stats['has_trend']}
Seasonality detected: {stats['has_seasonality']}
Stationary: {stats['stationarity']}

Recommend the best forecasting approaches. Return JSON only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            model_name = get_model_name(self.model_type)
            response = call_openrouter(
                messages=messages,
                model_name=model_name,
                api_key=self.api_key,
                function_name="analyze_timeseries"
            )

            result = self._extract_json(response)
            return result

        except Exception as e:
            # Fallback recommendations
            return self._fallback_recommendations(stats)

    def _extract_json(self, text):
        """Extract JSON from AI response"""
        try:
            # Try direct parsing
            return json.loads(text)
        except:
            # Try to find JSON in text
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("Could not extract JSON from response")

    def _fallback_column_detection(self, dataframe):
        """Fallback method to detect columns using heuristics"""

        date_column = None
        value_column = None

        # Look for date column
        for col in dataframe.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                date_column = col
                break

        # Try to parse as datetime
        if not date_column:
            for col in dataframe.columns:
                try:
                    pd.to_datetime(dataframe[col], errors='raise')
                    date_column = col
                    break
                except:
                    continue

        # Look for numeric value column
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Prefer columns with fewer nulls
            value_column = min(numeric_cols, key=lambda c: dataframe[c].isna().sum())

        return {
            'date_column': date_column or dataframe.columns[0],
            'value_column': value_column or dataframe.columns[1] if len(dataframe.columns) > 1 else dataframe.columns[0],
            'reasoning': 'Detected using fallback heuristics'
        }

    def _detect_trend(self, ts_data):
        """Detect if series has trend"""
        try:
            if len(ts_data) < 10:
                return False

            # Simple linear regression
            x = np.arange(len(ts_data))
            y = ts_data.values

            # Calculate correlation
            correlation = np.corrcoef(x, y)[0, 1]

            return abs(correlation) > 0.3
        except:
            return False

    def _detect_seasonality(self, ts_data):
        """Detect if series has seasonality"""
        try:
            if len(ts_data) < 24:
                return False

            # Check for repeating patterns using autocorrelation
            # Simplified check
            rolling_mean = ts_data.rolling(window=12).mean()
            rolling_std = ts_data.rolling(window=12).std()

            if rolling_std.dropna().std() > 0:
                return True

            return False
        except:
            return False

    def _check_stationarity(self, ts_data):
        """Check if series is stationary"""
        try:
            if len(ts_data) < 20:
                return "Unknown"

            # Split into two halves
            mid = len(ts_data) // 2
            first_half = ts_data[:mid]
            second_half = ts_data[mid:]

            # Compare means and stds
            mean_diff = abs(first_half.mean() - second_half.mean())
            std_diff = abs(first_half.std() - second_half.std())

            # If differences are small relative to overall stats
            if mean_diff < 0.3 * ts_data.std() and std_diff < 0.3 * ts_data.std():
                return "Likely Stationary"
            else:
                return "Non-Stationary"
        except:
            return "Unknown"

    def _fallback_recommendations(self, stats):
        """Fallback recommendations based on stats"""

        patterns = []
        methods = ['ARIMA', 'Exponential Smoothing', 'Trend Model']

        if stats['has_trend']:
            patterns.append("Trend detected")

        if stats['has_seasonality']:
            patterns.append("Seasonality detected")
            methods.insert(0, 'SARIMA')

        if not stats['has_trend'] and not stats['has_seasonality']:
            patterns.append("Random walk / No clear pattern")

        # Recommend horizon based on data length
        horizon = min(30, max(5, int(stats['length'] * 0.2)))

        return {
            'patterns': patterns if patterns else ["No strong patterns detected"],
            'recommended_methods': methods,
            'forecast_horizon': horizon,
            'reasoning': 'Recommendations based on statistical analysis'
        }
