"""
Analysis Fallback Handler

Provides intelligent fallback mechanisms when analysis cannot proceed due to data issues.
"""

import streamlit as st
from src.llm_service import call_openrouter, get_model_name


def suggest_alternative_analysis(error_type, target_variable, data_summary, api_key, model_type=4, user_query=None):
    """
    Use AI to suggest alternative analysis approaches when the current one fails.

    Args:
        error_type: Type of error (e.g., 'continuous_target_for_classification', 'insufficient_data')
        target_variable: Name of the target variable
        data_summary: Summary of the dataset
        api_key: OpenRouter API key
        model_type: Model type to use
        user_query: Original user question (optional)

    Returns:
        Dictionary with suggestion and next steps
    """

    error_descriptions = {
        'continuous_target_for_classification': f"The target variable '{target_variable}' appears to be continuous (numerical), but you selected Predictive Classification which requires categorical data.",
        'categorical_target_for_regression': f"The target variable '{target_variable}' appears to be categorical, but you selected Regression which requires continuous numerical data.",
        'insufficient_data': "The dataset has insufficient data for the selected analysis.",
        'invalid_target': f"The target variable '{target_variable}' is not suitable for this analysis.",
    }

    error_desc = error_descriptions.get(error_type, "The analysis cannot proceed with the current data.")

    system_prompt = """You are a helpful data science assistant helping users troubleshoot analysis issues.

When analysis fails, your job is to:
1. Explain the issue in simple terms
2. Suggest the most appropriate alternative analysis mode
3. Explain why that alternative would work better
4. Provide clear next steps for the user

Available analysis modes:
- Predictive Classification: For categorical target variables (e.g., yes/no, categories)
- Regression Model: For continuous numerical target variables (e.g., price, temperature)
- Clustering Model: For finding patterns without a specific target variable
- Data Visualization: For exploring the data

Be encouraging and helpful. Focus on guiding the user to success."""

    user_context = f"\n\nUser's Original Goal: {user_query}" if user_query else ""

    user_message = f"""A user encountered this issue:

{error_desc}

Dataset Summary:
{data_summary}

Target Variable: {target_variable}{user_context}

Please provide:
1. A clear explanation of why the analysis failed
2. The recommended analysis mode to use instead
3. Why that mode is more appropriate
4. Step-by-step instructions on what to do next

Format your response in a friendly, conversational way."""

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
            function_name="suggest_alternative_analysis"
        )
        return response
    except Exception as e:
        return None


def handle_analysis_error_with_fallback(error_type, target_variable, dataframe, api_key, model_type=4, user_query=None):
    """
    Display error and provide AI-powered alternative suggestions.

    Args:
        error_type: Type of error encountered
        target_variable: Name of the target variable
        dataframe: The dataset
        api_key: OpenRouter API key
        model_type: Model type to use
        user_query: Original user question
    """

    st.error("⚠️ Analysis Cannot Proceed")

    # Get dataset summary
    num_rows, num_cols = dataframe.shape
    target_type = str(dataframe[target_variable].dtype)
    unique_count = dataframe[target_variable].nunique()

    data_summary = f"""
    - Dataset size: {num_rows} rows × {num_cols} columns
    - Target variable: {target_variable}
    - Target data type: {target_type}
    - Unique values in target: {unique_count}
    """

    st.write("### 🤔 What Went Wrong?")

    with st.spinner("AI is analyzing the issue and finding alternatives..."):
        suggestion = suggest_alternative_analysis(
            error_type=error_type,
            target_variable=target_variable,
            data_summary=data_summary,
            api_key=api_key,
            model_type=model_type,
            user_query=user_query
        )

    if suggestion:
        st.markdown(suggestion)
    else:
        # Fallback guidance if AI fails
        st.write("The selected analysis mode is not compatible with your data.")

        if error_type == 'continuous_target_for_classification':
            st.info("""
            **The Issue:** Your target variable contains continuous numerical values, but Predictive Classification
            requires categorical data (like categories or yes/no values).

            **Recommended Solution:** Use **Regression Model** instead, which is designed for predicting continuous values.

            **Next Steps:**
            1. Go back to the analysis mode selector
            2. Choose "Regression Model"
            3. Start the analysis again
            """)

        elif error_type == 'categorical_target_for_regression':
            st.info("""
            **The Issue:** Your target variable contains categorical data, but Regression requires
            continuous numerical values.

            **Recommended Solution:** Use **Predictive Classification** instead, which is designed for categorical outcomes.

            **Next Steps:**
            1. Go back to the analysis mode selector
            2. Choose "Predictive Classification"
            3. Start the analysis again
            """)

    # Add a button to restart
    if st.button("🔄 Go Back and Choose Different Analysis Mode", type="primary"):
        # Clear the analysis state
        keys_to_clear = ['button_clicked', 'target_selected', 'data_origin', 'df_cleaned1',
                        'df_cleaned2', 'df_pca', 'balance_data', 'start_training', 'model_trained',
                        'decided_model', 'all_set']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
