# Changelog

All notable changes to AI Analytics Engine will be documented in this file.

## [2.2.0] - 2025-11-26

### Added - Time Series Forecasting & Enhanced Planning Agent

#### EDA-Powered Planning Agent
- **Data-Aware Routing** (`app/planning_agent.py`)
  - Added `perform_quick_eda()` - Analyzes dataset before routing
  - Added `analyze_query_with_data()` - AI considers both user intent AND data structure
  - Detects column types (numeric, categorical, datetime)
  - Calculates statistics (mean, std, min, max, variance)
  - Identifies missing values and unique counts
  - Smart datetime detection for time series routing
  - Provides `data_insights` explaining routing decisions

- **Enhanced Chat Interface** (`app/app.py`)
  - Planning agent now performs EDA before recommending mode
  - Displays data insights in conversation history
  - Better spinner message: "Analyzing your data and determining the best approach..."
  - Shows why AI chose specific analysis mode based on data characteristics

#### Time Series Forecasting Mode
- **New Time Series Pipeline** (`app/timeseries_model.py`)
  - Complete time series forecasting workflow with AI assistance
  - Multi-step pipeline: data inspection, preprocessing, analysis, forecasting
  - Support for multiple forecasting methods:
    - **ARIMA**: AutoRegressive Integrated Moving Average
    - **Exponential Smoothing**: With trend and seasonality support
    - **Trend Models**: ML-based (Random Forest) with lag features
  - Automatic train/test split and model evaluation
  - Forecast visualization with confidence intervals
  - Downloadable forecast results in CSV format

- **Time Series Agent** (`app/src/timeseries_agent.py`)
  - AI-powered column detection for date/time and value columns
  - Automatic pattern detection (trend, seasonality, stationarity)
  - Intelligent method recommendation based on data characteristics
  - Fallback heuristics when AI is unavailable
  - Statistical analysis:
    - Trend detection using correlation
    - Seasonality detection using rolling statistics
    - Stationarity checking

#### AI Features for Time Series
- **Smart Preprocessing**
  - Automatic date parsing and validation
  - AI-guided missing value imputation
  - Handles irregular time series
  - Duplicate removal

- **AI Summary for Forecasts** (`app/src/ai_summarizer.py`)
  - Added `summarize_timeseries_results()` function
  - Context-aware forecast analysis
  - Explains forecast quality and reliability
  - Compares model performance
  - Provides actionable recommendations

#### Integration
- **App Integration** (`app/app.py`)
  - Added "Time Series Forecasting" to analysis mode dropdown
  - Integrated time series pipeline into main workflow
  - Updated planning agent to recognize time series queries

- **Planning Agent Updates** (`app/planning_agent.py`)
  - Recognizes time series forecasting requests
  - Keywords: "forecast", "predict future", "time series", "trends over time"
  - Recommends time series mode for temporal data

#### Metrics & Visualization
- **Evaluation Metrics**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)

- **Visualizations**:
  - Historical data plot
  - Forecast vs actual comparison
  - 95% confidence intervals
  - Model comparison charts

#### Smart Correlation Analysis
- **Intelligent Column Selection** (`app/src/plot.py`)
  - Added `select_correlation_columns()` - Filters out inappropriate columns
  - Removes ID-like columns (>95% unique values)
  - Removes constant columns (zero variance)
  - Prioritizes columns by variance (most informative first)
  - Limits to top 15 most relevant columns
  - Automatic title updates showing column filtering

### Fixed

#### Bug Fixes
- **PCA NaN Handling** (`app/src/pca.py`)
  - Fixed `ValueError: Input X contains NaN` in all PCA functions
  - Added NaN detection and mean imputation before PCA
  - Applied to `perform_pca()`, `perform_PCA_for_regression()`, and `perform_PCA_for_clustering()`

- **Session State Initialization** (`app/prediction_model.py`)
  - Fixed `AttributeError: st.session_state has no attribute "to_perform_pca"`
  - Separated `to_perform_pca` initialization from `df_pca` check
  - Prevents crashes on Streamlit reruns

### Technical Details

#### New Dependencies
- Already available: `statsmodels==0.14.0` (ARIMA, SARIMAX, Exponential Smoothing)
- Uses existing: `scikit-learn` for ML-based forecasting

#### File Structure
```
app/
├── timeseries_model.py           # NEW: Time series forecasting pipeline
├── src/
│   ├── timeseries_agent.py       # NEW: AI agent for time series
│   ├── ai_summarizer.py          # MODIFIED: Added summarize_timeseries_results()
│   ├── plot.py                   # MODIFIED: Smart correlation column selection
│   └── pca.py                    # MODIFIED: NaN handling in all PCA functions
├── app.py                         # MODIFIED: EDA-powered planning agent
├── planning_agent.py              # MODIFIED: Data-aware routing with EDA
└── prediction_model.py            # MODIFIED: Fixed session state initialization
```

---

## [2.1.0] - 2025-11-24

### Added - Planning Agent & UX Improvements

#### AI Planning Assistant
- **New Planning Agent** (`app/planning_agent.py`)
  - Chat interface for users to ask questions about their data analysis needs
  - Analyzes user queries to determine the best analysis mode
  - Autonomous execution: automatically starts analysis when appropriate
  - Handles both data analysis and general questions
  - Stores conversation history for context-aware responses

- **Chat Interface** (`app/app.py`)
  - Added "Ask the Planning Assistant" section at top of page
  - Text input with "Ask Assistant" button
  - Displays last 5 conversations in expandable history
  - Shows recommended analysis mode and next steps
  - Auto-hides "Let's Get Started" section when chatting begins (preserves API key)

#### Context-Aware AI Summaries
- **New AI Summarizer Module** (`app/src/ai_summarizer.py`)
  - `summarize_prediction_results()` - Summarizes classification results
  - `summarize_regression_results()` - Summarizes regression results
  - `summarize_clustering_results()` - Summarizes clustering results
  - `summarize_visualization_insights()` - Summarizes data visualization findings
  - All functions accept `user_query` parameter to relate findings to original question
  - System prompts prioritize answering user's question first

- **Integration Across All Pipelines**
  - Added AI summaries to prediction, regression, clustering, and visualization pipelines
  - Summaries appear after results with 🤖 icon
  - User query stored in session state and passed to summarizers
  - Provides 4-7 actionable bullet points tailored to user's goals

#### Smart Fallback Mechanism
- **New Fallback Handler** (`app/src/analysis_fallback.py`)
  - `suggest_alternative_analysis()` - AI-powered alternative suggestions
  - `handle_analysis_error_with_fallback()` - Displays errors with helpful guidance
  - Handles common errors:
    - `continuous_target_for_classification` - Target is numeric but classification selected
    - `categorical_target_for_regression` - Target is categorical but regression selected
    - `insufficient_data` - Dataset too small for analysis
    - `invalid_target` - Target variable not suitable
  - "Go Back and Choose Different Analysis Mode" button to restart

- **Error Handling Updates** (`app/src/model_service.py`)
  - Modified `balance_target_classes()` to set error flag instead of stopping
  - Sets `st.session_state.analysis_error` for fallback mechanism
  - Graceful error handling without breaking workflow

#### UX Improvements
- **Download Buttons Repositioned**
  - Moved from main pipeline to inside `display_results()` functions
  - Now appear BEFORE AI Summary section (previously after)
  - Applied to:
    - `app/prediction_model.py` - Lines 324-357
    - `app/regression_model.py` - Lines 307-340
    - `app/cluster_model.py` - Lines 278-305
  - Consistent 3-column layout with download buttons for each model

- **Hidden Setup Section**
  - "Let's Get Started" section auto-hides when users start chatting
  - API key preserved in session state (`st.session_state.temp_api_key`)
  - Cleaner interface during chat interactions

#### API & Session Management
- **API Key Persistence**
  - Stored in `st.session_state.temp_api_key` across reruns
  - Used as default value in text input field
  - Prevents loss of API key during autonomous execution
  - Works correctly when planning agent triggers analysis

- **User Query Tracking**
  - Stored in `st.session_state.user_query` when planning agent executes
  - Passed to all AI summarizers for context-aware responses
  - Preserved throughout analysis workflow

### Changed

#### Function Signatures Updated
- **`display_results()` functions now accept additional parameters:**
  - Prediction: `display_results(X_train, X_test, Y_train, Y_test, target_variable, API_KEY, GPT_MODEL)`
  - Regression: `display_results(X_train, X_test, Y_train, Y_test, target_variable, API_KEY, GPT_MODEL)`
  - Clustering: `display_results(X, API_KEY, GPT_MODEL)`
  - Added to pass API key and model type for AI summarization

#### LLM Service Updates (`app/src/llm_service.py`)
- Planning agent uses `call_openrouter()` with `messages` format
- Added function logging for planning agent calls
- Model name resolution via `get_model_name()`

#### Visualization Pipeline (`app/visualization.py`)
- Added AI summary section at end of visualization
- Integrated `summarize_visualization_insights()`
- Passes user query for context-aware insights

### Technical Details

#### New Dependencies
- No new external dependencies (uses existing OpenRouter integration)

#### File Structure
```
app/
├── planning_agent.py              # NEW: Planning agent class
├── src/
│   ├── ai_summarizer.py          # NEW: AI summary functions
│   ├── analysis_fallback.py      # NEW: Fallback error handling
│   ├── llm_service.py            # MODIFIED: Added planning agent support
│   └── model_service.py          # MODIFIED: Improved error handling
├── app.py                         # MODIFIED: Chat interface & auto-hide logic
├── prediction_model.py            # MODIFIED: Download buttons & AI summary
├── regression_model.py            # MODIFIED: Download buttons & AI summary
├── cluster_model.py               # MODIFIED: Download buttons & AI summary
└── visualization.py               # MODIFIED: AI summary integration
```

#### Key Code Patterns

**Planning Agent Execution Flow:**
```python
1. User asks question → Planning agent analyzes
2. Agent sets should_execute = True if ready
3. app.py detects auto_execute flag
4. Stores user_query in session state
5. Triggers analysis with st.rerun()
6. Analysis runs with stored user_query
7. AI summary relates to user's question
```

**Fallback Flow:**
```python
1. Analysis encounters error (e.g., wrong data type)
2. Sets st.session_state.analysis_error flag
3. Main pipeline detects error
4. Calls handle_analysis_error_with_fallback()
5. AI suggests alternative mode
6. User clicks "Go Back" to restart
```

### Bug Fixes
- Fixed API key not persisting on autonomous execution rerun
- Fixed function signature mismatches in display_results()
- Fixed error handling to use fallback instead of hard stops
- Fixed download buttons appearing after summaries

---

## [2.0.0] - 2025-11-23

### Added - OpenRouter Migration

#### OpenRouter Integration
- Migrated from OpenAI API to OpenRouter API
- Added support for free models:
  - **Grok-4.1-Fast**: High-quality analysis (completely free)
  - **GPT-OSS-20B**: Fast processing (completely free)
- API key entry through secure UI password field
- No more environment variables or .env files

#### AI Logging System
- Created comprehensive AI call logging (`app/src/llm_service.py`)
- All AI interactions logged to `app/logs/ai_calls_YYYYMMDD.log`
- Logs include:
  - Function name
  - Model used
  - Timestamp
  - Full prompt
  - Full response
  - Errors (if any)
- Daily log files with automatic date handling
- Created log viewer utility (`app/view_ai_logs.py`)
  - `--count`: Show call statistics by function
  - `--latest`: Display most recent AI call
  - `--function`: Filter by specific function

#### Rebranding
- Updated application name from "Streamline Analyst" to "AI Analytics Engine"
- Updated all UI text, headers, and messages
- Updated page config and branding elements

### Changed

#### LLM Service Refactor (`app/src/llm_service.py`)
- Removed LangChain dependency (90% performance improvement)
- Direct API calls to OpenRouter
- Standardized `call_openrouter()` function
- Model name mapping: `get_model_name()`
- Enhanced error handling and retries

#### API Key Management
- Removed `.env` file dependency
- API key input through Streamlit UI
- Session state storage for key persistence
- Never saved to disk or git

### Fixed

#### Critical Bug Fixes
- **KeyError with Non-Numeric Columns**
  - Fixed `decide_balance_strategy()` to handle non-numeric target variables
  - Added proper column type checking before computing class counts
  - Better error messages for data type mismatches

- **Data Type Handling**
  - Improved detection of numeric vs categorical columns
  - Better preprocessing for mixed data types
  - Enhanced validation before model training

- **Error Messages**
  - More descriptive error messages throughout
  - Better user guidance when errors occur
  - Informative warnings for data issues

### Performance
- Removed LangChain: ~90% faster execution
- Direct API calls reduce overhead
- Reduced memory footprint
- Faster model selection and training

---

## [1.0.0] - Original Streamline Analyst

Created by [Zhe Lin](https://github.com/Wilson-ZheLin)

### Features
- OpenAI GPT-3.5 and GPT-4 integration
- Automated data preprocessing
- LLM-based model selection
- Support for classification, regression, clustering
- Data visualization tools
- Model training and evaluation
- Download trained models

---

## Legend

- **Added**: New features or functionality
- **Changed**: Changes to existing functionality
- **Fixed**: Bug fixes
- **Removed**: Removed features or functionality
- **Performance**: Performance improvements
- **Security**: Security improvements
