import os
import yaml
import json
import re
import streamlit as st
import requests
import logging
from datetime import datetime

config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# OpenRouter configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Configure logging for AI calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional: Save AI logs to a file
ai_log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(ai_log_dir, exist_ok=True)
ai_log_file = os.path.join(ai_log_dir, f'ai_calls_{datetime.now().strftime("%Y%m%d")}.log')

file_handler = logging.FileHandler(ai_log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def log_ai_interaction(function_name, model_name, prompt, response, error=None):
    """
    Log AI interaction details including prompt sent and response received.

    Parameters:
    - function_name: Name of the calling function
    - model_name: The AI model being used
    - prompt: The prompt sent to the AI
    - response: The response received from the AI
    - error: Any error that occurred (optional)
    """
    log_separator = "=" * 80

    if error:
        logger.error(f"\n{log_separator}")
        logger.error(f"AI CALL FAILED - Function: {function_name}")
        logger.error(f"Model: {model_name}")
        logger.error(f"Error: {error}")
        logger.error(f"Prompt sent:\n{prompt}")
        logger.error(f"{log_separator}\n")
    else:
        logger.info(f"\n{log_separator}")
        logger.info(f"AI CALL - Function: {function_name}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\nPrompt sent:\n{prompt}")
        logger.info(f"\nResponse received:\n{response}")
        logger.info(f"{log_separator}\n")

def get_api_key(user_api_key=None):
    """Get API key from user input only"""
    if not user_api_key:
        st.error("⚠️ API key is required. Please enter your OpenRouter API key in the input field above.")
        st.info("Get your free API key at: https://openrouter.ai/")
        st.stop()
    return user_api_key

def get_model_name(model_type=4):
    """Get OpenRouter model name based on type"""
    if model_type == 4:
        # GPT-4 equivalent - using Grok (high-quality, free)
        return "x-ai/grok-4.1-fast:free"
    else:
        # GPT-3.5 equivalent - using GPT-OSS (fast, free)
        return "openai/gpt-oss-20b:free"

def call_openrouter(messages, model_name, api_key, function_name="unknown"):
    """Make API call to OpenRouter with detailed logging"""
    if not api_key:
        st.error("⚠️ OpenRouter API key not configured. Please enter your API key in the input field above.")
        st.info("Get your free API key at: https://openrouter.ai/")
        st.stop()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Analytics Engine"
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0
    }

    # Log the prompt being sent
    prompt_text = messages[-1]['content'] if messages else "No prompt"
    logger.info(f"Calling AI - Function: {function_name}, Model: {model_name}")

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_content = response.json()['choices'][0]['message']['content']

        # Log successful interaction
        log_ai_interaction(function_name, model_name, prompt_text, response_content)

        return response_content
    except requests.exceptions.RequestException as e:
        # Log failed interaction
        log_ai_interaction(function_name, model_name, prompt_text, None, error=str(e))
        st.error(f"Cannot access the OpenRouter API. Please check your API key or network connection. Error: {str(e)}")
        st.stop()

def extract_json_from_response(response_text):
    """Extract JSON from response, handling markdown code blocks"""
    if '```json' in response_text:
        match = re.search(r'```json\n(.*?)```', response_text, re.DOTALL)
        if match:
            return match.group(1)
    return response_text

def decide_encode_type(attributes, data_frame_head, model_type=4, user_api_key=None):
    """
    Decides the encoding type for given attributes using OpenRouter API.

    Parameters:
    - attributes (list): A list of attributes for which to decide the encoding type.
    - data_frame_head (DataFrame): The head of the DataFrame containing the attributes.
    - model_type (int, optional): Specifies the model to use (4 for GPT-4 equivalent, 3 for GPT-3.5 equivalent).
    - user_api_key (str, optional): The user's OpenRouter API key.

    Returns:
    - A JSON object containing the recommended encoding types for the given attributes.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)
        
        template = config["numeric_attribute_template"]
        prompt = template.format(attributes=attributes, data_frame_head=data_frame_head)
        
        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_encode_type")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error in decide_encode_type: {str(e)}")
        st.stop()

def decide_fill_null(attributes, types_info, description_info, model_type=4, user_api_key=None):
    """
    Decides the best method to fill null values using OpenRouter API.

    Parameters:
    - attributes (list): List of attribute names to consider.
    - types_info: Information about data types.
    - description_info: Descriptive statistics.
    - model_type (int, optional): The model to use.
    - user_api_key (str, optional): The user's OpenRouter API key.

    Returns:
    - dict: A JSON object with recommended null filling methods.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)
        
        template = config["null_attribute_template"]
        prompt = template.format(
            attributes=attributes,
            types_info=types_info,
            description_info=description_info
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_fill_null")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error in decide_fill_null: {str(e)}")
        st.stop()

def decide_model(shape_info, head_info, nunique_info, description_info, model_type=4, user_api_key=None):
    """
    Decides the most suitable machine learning model based on dataset characteristics.

    Parameters:
    - shape_info (dict): Information about the shape of the dataset.
    - head_info (str or DataFrame): The head of the dataset.
    - nunique_info (dict): Information about the uniqueness of dataset attributes.
    - description_info (str): Descriptive information about the dataset.
    - model_type (int, optional): Specifies which model to consult.
    - user_api_key (str, optional): OpenRouter API key.

    Returns:
    - dict: A JSON object containing the recommended model and configuration.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)

        template = config["decide_model_template"]
        prompt = template.format(
            shape_info=shape_info,
            head_info=head_info,
            nunique_info=nunique_info,
            description_info=description_info
        )

        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_model")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error in decide_model: {str(e)}")
        st.stop()

def decide_cluster_model(shape_info, description_info, cluster_info, model_type=4, user_api_key=None):
    """
    Determines the appropriate clustering model based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - cluster_info: Additional information relevant to clustering.
    - model_type (int, optional): The model type to use (default 4).
    - user_api_key (str, optional): The user's API key for OpenRouter.

    Returns:
    - A JSON object with the recommended clustering model and parameters.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)

        template = config["decide_clustering_model_template"]
        prompt = template.format(
            shape_info=shape_info,
            description_info=description_info,
            cluster_info=cluster_info
        )

        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_cluster_model")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error in decide_cluster_model: {str(e)}")
        st.stop()

def decide_regression_model(shape_info, description_info, Y_name, model_type=4, user_api_key=None):
    """
    Determines the appropriate regression model based on dataset characteristics and target variable.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - Y_name: The name of the target variable.
    - model_type (int, optional): The model type to use (default 4).
    - user_api_key (str, optional): The user's API key for OpenRouter.

    Returns:
    - A JSON object with the recommended regression model and parameters.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)

        template = config["decide_regression_model_template"]
        prompt = template.format(
            shape_info=shape_info,
            description_info=description_info,
            Y_name=Y_name
        )

        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_regression_model")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error in decide_regression_model: {str(e)}")
        st.stop()

def decide_target_attribute(attributes, types_info, head_info, model_type=4, user_api_key=None):
    """
    Determines the target attribute for modeling based on dataset attributes and characteristics.

    Parameters:
    - attributes: A list of dataset attributes.
    - types_info: Information about the data types of the attributes.
    - head_info: A snapshot of the dataset's first few rows.
    - model_type (int, optional): The model type to use (default 4).
    - user_api_key (str, optional): The user's API key for OpenRouter.

    Returns:
    - The name of the recommended target attribute.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)

        template = config["decide_target_attribute_template"]
        prompt = template.format(
            attributes=attributes,
            types_info=types_info,
            head_info=head_info
        )

        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_target_attribute")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)["target"]
    except Exception as e:
        st.error(f"Error in decide_target_attribute: {str(e)}")
        st.stop()

def decide_test_ratio(shape_info, model_type=4, user_api_key=None):
    """
    Determines the appropriate train-test split ratio based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - model_type (int, optional): The model type to use (default 4).
    - user_api_key (str, optional): The user's API key for OpenRouter.

    Returns:
    - The recommended train-test split ratio as a float.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)

        template = config["decide_test_ratio_template"]
        prompt = template.format(shape_info=shape_info)

        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_test_ratio")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)["test_ratio"]
    except Exception as e:
        st.error(f"Error in decide_test_ratio: {str(e)}")
        st.stop()

def decide_balance(shape_info, description_info, balance_info, model_type=4, user_api_key=None):
    """
    Determines the appropriate method to balance the dataset based on its characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - balance_info: Additional information relevant to dataset balancing.
    - model_type (int, optional): The model type to use (default 4).
    - user_api_key (str, optional): The user's API key for OpenRouter.

    Returns:
    - The recommended method to balance the dataset.
    """
    try:
        model_name = get_model_name(model_type)
        api_key = get_api_key(user_api_key)

        template = config["decide_balance_template"]
        prompt = template.format(
            shape_info=shape_info,
            description_info=description_info,
            balance_info=balance_info
        )

        messages = [{"role": "user", "content": prompt}]
        response = call_openrouter(messages, model_name, api_key, function_name="decide_balance")

        json_str = extract_json_from_response(response)
        return json.loads(json_str)["method"]
    except Exception as e:
        st.error(f"Error in decide_balance: {str(e)}")
        st.stop()