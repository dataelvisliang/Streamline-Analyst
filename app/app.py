import time
import streamlit as st
from streamlit_lottie import st_lottie
from util import load_lottie, stream_data, welcome_message, introduction_message
from prediction_model import prediction_model_pipeline
from cluster_model import cluster_model_pipeline
from regression_model import regression_model_pipeline
from visualization import data_visualization
from src.util import read_file_from_streamlit

st.set_page_config(page_title="AI Analytics Engine", page_icon=":rocket:", layout="wide")

# TITLE SECTION
with st.container():
    st.subheader("Hello there 👋")
    st.title("Welcome to AI Analytics Engine!")
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    if st.session_state.initialized:
        st.session_state.welcome_message = welcome_message()
        st.write(stream_data(st.session_state.welcome_message))
        time.sleep(0.5)
        st.write("[Github > ](https://github.com/Wilson-ZheLin/Streamline-Analyst)")
        st.session_state.initialized = False
    else:
        st.write(st.session_state.welcome_message)
        st.write("[Github > ](https://github.com/Wilson-ZheLin/Streamline-Analyst)")

# INTRO SECTION
with st.container():
    st.divider()
    if 'lottie' not in st.session_state:
        st.session_state.lottie_url1, st.session_state.lottie_url2 = load_lottie()
        st.session_state.lottie = True

    left_column_r1, right_column_r1 = st.columns([6, 4])
    with left_column_r1:
        st.header("What can AI Analytics Engine do?")
        st.write(introduction_message()[0])
    with right_column_r1:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url1, height=280, key="animation1")

    left_column_r2, _, right_column_r2 = st.columns([6, 1, 5])
    with left_column_r2:
        if st.session_state.lottie:
            st_lottie(st.session_state.lottie_url2, height=200, key="animation2")
    with right_column_r2:
        st.header("Simple to Use")
        st.write(introduction_message()[1])

# MAIN SECTION
with st.container():
    st.divider()
    st.header("Let's Get Started")
    left_column, right_column = st.columns([6, 4])
    with left_column:
        API_KEY = st.text_input(
            "Your OpenRouter API Key",
            placeholder="Enter your OpenRouter API key here... (e.g., sk-or-v1-...)",
            type="password",
            help="Get your free API key at https://openrouter.ai/"
        )
        if not API_KEY:
            st.caption("🔑 Your API Key won't be stored or shared!")
            st.info("📌 **Don't have an API key?** Get one for free at [OpenRouter.ai](https://openrouter.ai/)")
        else:
            st.caption("✅ API key provided")
        
        uploaded_file = st.file_uploader(
            "Choose a data file. Your data won't be stored as well!", 
            accept_multiple_files=False, 
            type=['csv', 'json', 'xls', 'xlsx']
        )
        if uploaded_file:
            if uploaded_file.getvalue():
                uploaded_file.seek(0)
                st.session_state.DF_uploaded = read_file_from_streamlit(uploaded_file)
                st.session_state.is_file_empty = False
            else:
                st.session_state.is_file_empty = True
        
    with right_column:
        SELECTED_MODEL = st.selectbox(
            'Which AI model do you want to use?',
            ('Grok-4.1-Fast (Free)', 'GPT-OSS-20B (Free)'),
            help="Both models are completely free! Grok-4.1 is higher quality, GPT-OSS is faster."
        )

        MODE = st.selectbox(
            'Select proper data analysis mode',
            ('Predictive Classification', 'Clustering Model', 'Regression Model', 'Data Visualization'),
            help="Data Visualization mode doesn't require an API key"
        )
        
        st.write(f'Model selected: :green[{SELECTED_MODEL}]')
        st.write(f'Data analysis mode: :green[{MODE}]')

    # Proceed Button
    is_proceed_enabled = uploaded_file is not None and API_KEY != "" or uploaded_file is not None and MODE == "Data Visualization"

    # Initialize the 'button_clicked' state
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    if st.button('Start Analysis', disabled=(not is_proceed_enabled) or st.session_state.button_clicked, type="primary"):
        st.session_state.button_clicked = True
    if "is_file_empty" in st.session_state and st.session_state.is_file_empty:
        st.caption('Your data file is empty!')

    # Start Analysis
    if st.session_state.button_clicked:
        GPT_MODEL = 4 if SELECTED_MODEL == 'Grok-4.1-Fast (Free)' else 3.5
        with st.container():
            if "DF_uploaded" not in st.session_state:
                st.error("File is empty!")
            else:
                if MODE == 'Predictive Classification':
                    prediction_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == 'Clustering Model':
                    cluster_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == 'Regression Model':
                    regression_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == 'Data Visualization':
                    data_visualization(st.session_state.DF_uploaded)

# python -m streamlit run app.py