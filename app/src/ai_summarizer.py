"""
AI Summarizer Module

This module provides functions to generate AI-powered summaries of data analysis results.
"""

import streamlit as st
from src.llm_service import call_openrouter, get_model_name


def summarize_prediction_results(target_variable, model_names, accuracies, f1_scores, is_binary, api_key, model_type=4, user_query=None):
    """
    Generate an AI summary of predictive classification results.

    Args:
        target_variable: Name of the target variable being predicted
        model_names: List of model names used
        accuracies: List of accuracy scores for each model
        f1_scores: List of F1 scores for each model
        is_binary: Whether this is binary classification
        api_key: OpenRouter API key
        model_type: Model type to use (default 4)
        user_query: Original user question/query (optional)

    Returns:
        String summary of the analysis
    """

    # Build results summary
    results_text = f"Target Variable: {target_variable}\n"
    results_text += f"Classification Type: {'Binary' if is_binary else 'Multi-class'}\n\n"
    results_text += "Model Performance:\n"

    for i, (name, acc, f1) in enumerate(zip(model_names, accuracies, f1_scores), 1):
        results_text += f"{i}. {name}\n"
        results_text += f"   - Accuracy: {acc:.4f}\n"
        results_text += f"   - F1 Score: {f1:.4f}\n"

    system_prompt = """You are a data science expert analyzing machine learning results.
Your task is to summarize the key findings from a predictive classification analysis in a clear, concise way.

Provide:
1. Direct answer to the user's original question (if provided)
2. Overall assessment of model performance
3. Which model performed best and why
4. Key insights about the predictive task and what they mean for the user's goal
5. 2-3 actionable recommendations

Keep your summary to 5-7 bullet points. Be specific, data-driven, and focus on answering the user's question."""

    user_context = f"\n\nUser's Original Question: {user_query}" if user_query else ""

    user_message = f"""Analyze these predictive classification results and provide a concise summary:

{results_text}{user_context}

Provide your analysis as bullet points. Start by directly addressing the user's question if provided."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        model_name = get_model_name(model_type)
        response = call_openrouter(
            messages=messages,
            model_name=model_name,
            api_key=api_key,
            function_name="summarize_prediction_results"
        )
        return response
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"


def summarize_clustering_results(n_clusters, silhouette_score, inertia, api_key, model_type=4, user_query=None):
    """
    Generate an AI summary of clustering results.

    Args:
        n_clusters: Number of clusters found
        silhouette_score: Silhouette score of the clustering
        inertia: Inertia value
        api_key: OpenRouter API key
        model_type: Model type to use
        user_query: Original user question/query (optional)

    Returns:
        String summary of the analysis
    """

    results_text = f"Number of Clusters: {n_clusters}\n"
    results_text += f"Silhouette Score: {silhouette_score:.4f}\n"
    results_text += f"Inertia: {inertia:.2f}\n"

    system_prompt = """You are a data science expert analyzing clustering results.
Your task is to summarize the key findings from a clustering analysis in a clear, concise way.

Provide:
1. Direct answer to the user's original question (if provided)
2. Assessment of clustering quality based on the metrics
3. What the number of clusters suggests about the data and patterns
4. How these findings relate to the user's goal
5. 2-3 actionable recommendations for using these clusters

Keep your summary to 4-6 bullet points. Be specific, data-driven, and focus on answering the user's question."""

    user_context = f"\n\nUser's Original Question: {user_query}" if user_query else ""

    user_message = f"""Analyze these clustering results and provide a concise summary:

{results_text}{user_context}

Provide your analysis as bullet points. Start by directly addressing the user's question if provided."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        model_name = get_model_name(model_type)
        response = call_openrouter(
            messages=messages,
            model_name=model_name,
            api_key=api_key,
            function_name="summarize_clustering_results"
        )
        return response
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"


def summarize_regression_results(target_variable, model_names, r2_scores, mse_scores, api_key, model_type=4, user_query=None):
    """
    Generate an AI summary of regression results.

    Args:
        target_variable: Name of the target variable being predicted
        model_names: List of model names used
        r2_scores: List of R² scores for each model
        mse_scores: List of MSE scores for each model
        api_key: OpenRouter API key
        model_type: Model type to use
        user_query: Original user question/query (optional)

    Returns:
        String summary of the analysis
    """

    results_text = f"Target Variable: {target_variable}\n\n"
    results_text += "Model Performance:\n"

    for i, (name, r2, mse) in enumerate(zip(model_names, r2_scores, mse_scores), 1):
        results_text += f"{i}. {name}\n"
        results_text += f"   - R² Score: {r2:.4f}\n"
        results_text += f"   - MSE: {mse:.4f}\n"

    system_prompt = """You are a data science expert analyzing regression results.
Your task is to summarize the key findings from a regression analysis in a clear, concise way.

Provide:
1. Direct answer to the user's original question (if provided)
2. Overall assessment of model performance
3. Which model performed best and why
4. What the R² and MSE values indicate for the user's goal
5. 2-3 actionable recommendations

Keep your summary to 5-7 bullet points. Be specific, data-driven, and focus on answering the user's question."""

    user_context = f"\n\nUser's Original Question: {user_query}" if user_query else ""

    user_message = f"""Analyze these regression results and provide a concise summary:

{results_text}{user_context}

Provide your analysis as bullet points. Start by directly addressing the user's question if provided."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        model_name = get_model_name(model_type)
        response = call_openrouter(
            messages=messages,
            model_name=model_name,
            api_key=api_key,
            function_name="summarize_regression_results"
        )
        return response
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"


def summarize_visualization_insights(data_overview, num_features, num_categorical, api_key, model_type=4, user_query=None):
    """
    Generate an AI summary of data visualization insights.

    Args:
        data_overview: Overview of the dataset (shape, columns, etc.)
        num_features: Number of numerical features
        num_categorical: Number of categorical features
        api_key: OpenRouter API key
        model_type: Model type to use
        user_query: Original user question/query (optional)

    Returns:
        String summary of the analysis
    """

    results_text = f"Dataset Overview:\n{data_overview}\n\n"
    results_text += f"Number of Numerical Features: {num_features}\n"
    results_text += f"Number of Categorical Features: {num_categorical}\n"

    system_prompt = """You are a data science expert analyzing data visualizations.
Your task is to provide key insights from the dataset exploration in a clear, concise way.

Provide:
1. Direct answer to the user's original question (if provided)
2. Overview of the dataset characteristics
3. Notable patterns or distributions observed
4. Potential data quality issues to watch for
5. How these findings relate to the user's goal
6. 2-3 recommended next steps for analysis

Keep your summary to 5-7 bullet points. Be specific, actionable, and focus on answering the user's question."""

    user_context = f"\n\nUser's Original Question: {user_query}" if user_query else ""

    user_message = f"""Analyze this dataset and provide key insights from the visualizations:

{results_text}{user_context}

Provide your analysis as bullet points. Start by directly addressing the user's question if provided."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        model_name = get_model_name(model_type)
        response = call_openrouter(
            messages=messages,
            model_name=model_name,
            api_key=api_key,
            function_name="summarize_visualization_insights"
        )
        return response
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"


def summarize_timeseries_results(best_model, model_metrics, forecast_horizon, api_key, model_type=4, user_query=None):
    """
    Generate AI summary for time series forecasting results

    Args:
        best_model: Name of the best performing model
        model_metrics: Dictionary of model names and their metrics (RMSE, MAE, MSE)
        forecast_horizon: Number of periods forecasted
        api_key: OpenRouter API key
        model_type: Model type (4 = Grok-4.1-Fast, 3.5 = GPT-OSS-20B)
        user_query: User's original question (optional)

    Returns:
        AI-generated summary as markdown text
    """

    # Format results for AI
    results_text = f"Best Performing Model: {best_model}\n\n"
    results_text += f"Forecast Horizon: {forecast_horizon} periods\n\n"
    results_text += "Model Performance Metrics:\n"

    for model_name, metrics in model_metrics.items():
        results_text += f"\n{model_name}:\n"
        results_text += f"  - RMSE: {metrics['RMSE']:.4f}\n"
        results_text += f"  - MAE: {metrics['MAE']:.4f}\n"
        results_text += f"  - MSE: {metrics['MSE']:.4f}\n"

    system_prompt = """You are a time series forecasting expert analyzing forecasting results.
Your task is to summarize the key findings from time series forecasting in a clear, actionable way.

Provide:
1. Direct answer to the user's original question (if provided)
2. Overall assessment of forecast quality and reliability
3. Why the best model performed better than others
4. What the error metrics (RMSE, MAE) indicate about forecast accuracy
5. Practical insights about the forecast (trends, patterns, confidence)
6. 2-3 actionable recommendations for using the forecast

Keep your summary to 5-7 bullet points. Be specific, data-driven, and focus on answering the user's question."""

    user_context = f"\n\nUser's Original Question: {user_query}" if user_query else ""

    user_message = f"""Analyze these time series forecasting results and provide a concise summary:

{results_text}{user_context}

Provide your analysis as bullet points. Start by directly addressing the user's question if provided."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        model_name = get_model_name(model_type)
        response = call_openrouter(
            messages=messages,
            model_name=model_name,
            api_key=api_key,
            function_name="summarize_timeseries_results"
        )
        return response
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"
