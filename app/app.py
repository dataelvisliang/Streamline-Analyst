import time
import streamlit as st
from util import stream_data, welcome_message
from prediction_model import prediction_model_pipeline
from cluster_model import cluster_model_pipeline
from regression_model import regression_model_pipeline
from visualization import data_visualization
from src.util import read_file_from_streamlit
from planning_agent import PlanningAgent

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

# CHAT INTERFACE WITH PLANNING AGENT
with st.container():
    st.divider()
    st.header("💬 Ask the Planning Assistant")
    st.write("Not sure which analysis mode to use? Chat with our AI planning assistant to help you decide!")

    # Initialize chat history and planning agent
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'planning_agent' not in st.session_state:
        st.session_state.planning_agent = None

    # Chat interface
    chat_col1, chat_col2 = st.columns([7, 3])

    with chat_col1:
        user_question = st.text_input(
            "Ask a question about your data analysis needs:",
            placeholder="e.g., 'I want to predict customer churn' or 'How do I find patterns in my sales data?'",
            key="user_question"
        )

    with chat_col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_button = st.button("Ask Assistant", type="secondary")

    if ask_button and user_question:
        # Get API key and initialize planning agent if needed
        temp_api_key = st.session_state.get('temp_api_key', '')
        if not temp_api_key:
            st.warning("Please enter your API key below to use the planning assistant.")
        else:
            if st.session_state.planning_agent is None:
                model_type = st.session_state.get('selected_model_type', 4)
                st.session_state.planning_agent = PlanningAgent(temp_api_key, model_type)

            # Check if data is uploaded
            has_data = 'DF_uploaded' in st.session_state and st.session_state.DF_uploaded is not None

            # Get response from planning agent
            with st.spinner("Thinking..."):
                analysis = st.session_state.planning_agent.analyze_query(user_question, has_data)

            # Add to chat history
            st.session_state.chat_history.append({
                "user": user_question,
                "assistant": analysis['response'],
                "analysis_mode": analysis.get('analysis_mode'),
                "next_steps": analysis.get('next_steps', [])
            })

            # Trigger autonomous execution if planning agent recommends it
            if st.session_state.planning_agent.should_execute and has_data:
                st.session_state.auto_execute = True
                st.session_state.auto_mode = st.session_state.planning_agent.execution_mode
                st.session_state.user_query = user_question  # Store user query for AI summary
                st.session_state.planning_agent.reset_execution_flags()
                st.rerun()

    # Display chat history
    if st.session_state.chat_history:
        st.write("---")
        st.subheader("Conversation History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q: {chat['user'][:50]}...", expanded=(i==0)):
                st.write(f"**You:** {chat['user']}")
                st.write(f"**Assistant:** {chat['assistant']}")
                if chat.get('analysis_mode'):
                    st.info(f"📊 Recommended Mode: **{chat['analysis_mode']}**")
                if chat.get('next_steps'):
                    st.write("**Next Steps:**")
                    for step in chat['next_steps']:
                        st.write(f"- {step}")

# MAIN SECTION
with st.container():
    st.divider()

    # Hide "Let's Get Started" section if user has started chatting
    show_setup_section = len(st.session_state.get('chat_history', [])) == 0

    if show_setup_section:
        st.header("Let's Get Started")
        left_column, right_column = st.columns([6, 4])
        with left_column:
            # Initialize API key from session state if it exists
            default_api_key = st.session_state.get('temp_api_key', '')

            API_KEY = st.text_input(
                "Your OpenRouter API Key",
                placeholder="Enter your OpenRouter API key here... (e.g., sk-or-v1-...)",
                type="password",
                help="Get your free API key at https://openrouter.ai/",
                value=default_api_key if default_api_key else ""
            )
            # Store API key in session state for planning agent and rerun persistence
            if API_KEY:
                st.session_state.temp_api_key = API_KEY

            if not API_KEY and not default_api_key:
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
            # Store selected model type for planning agent
            st.session_state.selected_model_type = 4 if SELECTED_MODEL == 'Grok-4.1-Fast (Free)' else 3.5

            MODE = st.selectbox(
                'Select proper data analysis mode',
                ('Predictive Classification', 'Clustering Model', 'Regression Model', 'Data Visualization'),
                help="Data Visualization mode doesn't require an API key"
            )

            st.write(f'Model selected: :green[{SELECTED_MODEL}]')
            st.write(f'Data analysis mode: :green[{MODE}]')

        # Proceed Button
        is_proceed_enabled = uploaded_file is not None and API_KEY != "" or uploaded_file is not None and MODE == "Data Visualization"
    else:
        # Section is hidden, but preserve API key and set default values
        API_KEY = st.session_state.get('temp_api_key', '')
        uploaded_file = None
        SELECTED_MODEL = 'Grok-4.1-Fast (Free)'
        MODE = 'Predictive Classification'
        is_proceed_enabled = False

    # Check for autonomous execution trigger
    if 'auto_execute' not in st.session_state:
        st.session_state.auto_execute = False
    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = None

    # If planning agent triggered autonomous execution
    if st.session_state.auto_execute and st.session_state.auto_mode:
        MODE = st.session_state.auto_mode
        st.info(f"🤖 Planning Assistant is starting **{MODE}** analysis automatically...")
        st.session_state.button_clicked = True
        st.session_state.auto_execute = False
        st.session_state.auto_mode = None

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
        # Use stored API key if current one is empty (happens on rerun)
        FINAL_API_KEY = API_KEY if API_KEY else st.session_state.get('temp_api_key', '')

        with st.container():
            if "DF_uploaded" not in st.session_state:
                st.error("File is empty!")
            else:
                if MODE == 'Predictive Classification':
                    prediction_model_pipeline(st.session_state.DF_uploaded, FINAL_API_KEY, GPT_MODEL)
                elif MODE == 'Clustering Model':
                    cluster_model_pipeline(st.session_state.DF_uploaded, FINAL_API_KEY, GPT_MODEL)
                elif MODE == 'Regression Model':
                    regression_model_pipeline(st.session_state.DF_uploaded, FINAL_API_KEY, GPT_MODEL)
                elif MODE == 'Data Visualization':
                    data_visualization(st.session_state.DF_uploaded, FINAL_API_KEY, GPT_MODEL)

# python -m streamlit run app.py