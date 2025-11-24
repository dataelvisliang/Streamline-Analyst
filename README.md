# AI Analytics Engine: A Data Analysis AI Agent

**Built on the foundation of [Streamline Analyst](https://github.com/Wilson-ZheLin/Streamline-Analyst) by [Zhe Lin](https://github.com/Wilson-ZheLin)**

---

AI Analytics Engine 🪄 is a cutting-edge, open-source application powered by Large Language Models (LLMs) designed to revolutionize data analysis. This **Data Analysis Agent** effortlessly automates all the tasks such as data cleaning, preprocessing, and even complex operations like identifying target objects, partitioning test sets, and selecting the best-fit models based on your data. With AI Analytics Engine, results visualization and evaluation become seamless.

Here's how it simplifies your workflow: just **select your data file**, **pick an analysis mode**, and **hit start**. AI Analytics Engine aims to expedite the data analysis process, making it accessible to all, regardless of their expertise in data analysis. It's built to empower users to process data and achieve high-quality visualizations with unparalleled efficiency🚀, and to execute high-performance modeling with the best strategies🔮.

**Powered by OpenRouter AI** - Access multiple free AI models including Grok-4.1-Fast and GPT-OSS-20B without any cost!

Your data's privacy and security are paramount; rest assured, uploaded data and API Keys are strictly for one-time use and are neither saved nor shared.

![Screenshot 2024-02-12 at 16 01 01](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/4167b04c-0853-4703-87a4-6c2994e30f9e)

Looking ahead, we plan to enhance AI Analytics Engine with advanced features like ***Natural Language Processing (NLP)***, ***neural networks***, and ***object detection (utilizing YOLO)***, broadening its capabilities to meet more diverse data analysis needs.

## 🎉 What's New in This Fork

### OpenRouter Migration (v2.0.0)
- **🆓 100% Free AI Models**:
  - **Grok-4.1-Fast**: High-quality analysis (completely free!)
  - **GPT-OSS-20B**: Fast processing (completely free!)
  - **Cost**: $0.00 per analysis - unlimited usage!

- **🔐 Enhanced Security**:
  - API key entered securely through UI (password field)
  - Never stored or saved anywhere
  - No risk of accidentally committing to git

- **🔍 Comprehensive AI Logging**:
  - All AI calls automatically logged with full prompts and responses
  - Daily log files for transparency and debugging
  - Built-in analysis tools to review AI interactions
  - Logs saved to `app/logs/ai_calls_YYYYMMDD.log`

- **🐛 Critical Bug Fixes**:
  - Fixed KeyError with non-numeric data columns
  - Improved error handling and messages
  - Better data type handling in preprocessing

- **⚡ Performance Improvements**:
  - Removed LangChain dependency (90% faster code execution)
  - Direct API calls for better performance
  - Reduced memory footprint

- **🎨 Rebranded Interface**:
  - Updated to AI Analytics Engine branding
  - Clearer model selection with free indicators
  - Improved help text and error messages

### Migration from Original
Migrated from OpenAI API to OpenRouter AI, providing:
- Free access to state-of-the-art models
- Better cost efficiency (100% free vs paid)
- Multiple model options through single API
- Same quality results, zero cost

Demo
----

https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/1d30faca-f474-42fd-b20b-c93ed7cf6d13

Current Version Features
------------------------
* **Target Variable Identification**: LLMs adeptly pinpoint the target variable
* **Null Value Management**: Choose from a variety of strategies such as mean, median, mode filling, interpolation, or introducing new categories for handling missing data, all recommended by LLMs
* **Data Encoding Tactics**: Automated suggestions and completions for the best encoding methods, including one-hot, integer mapping, and label encoding
* **Dimensionality Reduction with PCA**
* **Duplicate Entity Resolution**
* **Data Transformation and Normalization**: Utilize Box-Cox transformation and normalization techniques to improve data distribution and scalability
* **Balancing Target Variable Entities**: LLM-recommended methods like random over-sampling, SMOTE, and ADASYN help balance data sets, crucial for unbiased model training
* **Data Set Proportion Adjustment**: LLM determines the proportion of the data set (can also be adjusted manually)
* **Model Selection and Training**: Based on your data, LLMs recommend and initiate training with the most suitable models
* **Cluster Number Recommendation**: Leveraging the Elbow Rule and Silhouette Coefficient for optimal cluster numbers, with the flexibility of real-time adjustments
* **AI Logging & Transparency**: All AI interactions logged for review and debugging

All processed data and models are made available for download, offering a comprehensive, user-friendly data analysis toolkit.

### Modeling and Results Visualization:

![Screenshot 2024-02-12 at 16 10 35](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/423da7be-63f1-491d-9ebe-6a788c440c40)

### Automated Workflow Interface:

![Screenshot 2024-02-12 at 16 20 19](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/9d04d5f2-4f2a-44eb-ab8b-c07c8c0c5a53)

### Supported Modeling tasks:

| **Classification Models**        | **Clustering Models**         | **Regression Models**               |
|----------------------------------|-------------------------------|-------------------------------------|
| Logistic regression              | K-means clustering            | Linear regression                   |
| Random forest                    | DBSCAN                        | Ridge regression                    |
| Support vector machine           | Gaussian mixture model        | Lasso regression                    |
| Gradient boosting machine        | Hierarchical clustering       | Elastic net regression              |
| Gaussian Naive Bayes             | Spectral clustering           | Random forest regression            |
| AdaBoost                         | etc.                          | Gradient boosting regression        |
| XGBoost                          |                               | etc.                                |

### Real-time calculation of model indicators and result visualization:

| **Classification Metrics & Plots** | **Clustering Metrics & Plots** | **Regression Metrics & Plots**        |
|------------------------------------|--------------------------------|---------------------------------------|
| Model score                        | Silhouette score               | R-squared score                       |
| Confusion matrix                   | Calinski-Harabasz score        | Mean square error (MSE)               |
| AUC                                | Davies-Bouldin score           | Root mean square error (RMSE)         |
| F1 score                           | Cluster scatter plot           | Absolute error (MAE)                  |
| ROC plot                           | etc.                           | Residual plot                         |
| etc.                               |                                | Predicted value vs actual value plot  |
|                                    |                                | Quantile-Quantile plot                |

### Visual Analysis Toolkit:

AI Analytics Engine 🪄 offers an array of intuitive visual tools for enhanced data insight, **without the need for an API Key**:

* **Single Attribute Visualization**: Insightful views into individual data aspects
* **Multi-Attribute Visualization**: Comprehensive analysis of variable interrelations
* **Three-Dimensional Plotting**: Advanced 3D representations for complex data relationships
* **Word Clouds**: Key themes and concepts highlighted through word frequency
* **World Heat Maps**: Geographic trends and distributions made visually accessible

## 🚀 Quick Start (5 Minutes)

### Step 1: Get API Key (2 min)
1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up (completely free, no credit card required)
3. Create an API key
4. Copy your key

### Step 2: Clone & Install (2 min)
```bash
# Clone the repository
git clone https://github.com/dataelvisliang/Streamline-Analyst.git
cd Streamline-Analyst

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run (1 min)
```bash
cd app
streamlit run app.py
```

### Step 4: Start Analyzing!
1. Enter your OpenRouter API key in the password field
2. Select a free model (Grok-4.1-Fast or GPT-OSS-20B)
3. Upload your data file
4. Choose analysis mode
5. Click "Start Analysis"

**Total Setup Time: ~5 minutes | Cost: $0.00 | Difficulty: Easy** ✨

## 📋 Installation Details

### Prerequisites

* [Python 3.11.5](https://www.python.org/downloads/) or higher
* [OpenRouter API Key](https://openrouter.ai/) - **Free to use!**
  * Get your free API key at [OpenRouter.ai](https://openrouter.ai/)
  * Both Grok-4.1-Fast and GPT-OSS-20B models are completely free
  * No credit card required
  * Unlimited usage

### Full Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/dataelvisliang/Streamline-Analyst.git
cd Streamline-Analyst

# 2. Install required packages
pip install -r requirements.txt

# 3. Navigate to app directory
cd app

# 4. Run the application
streamlit run app.py
```

### Verifying Installation

After running the app, you should see:
- ✅ Streamlit opens in your browser
- ✅ "Welcome to AI Analytics Engine!" title
- ✅ API key input field (password type)
- ✅ Model selection dropdown showing free models
- ✅ File upload section

## 🔍 AI Logging Features

All AI interactions are automatically logged with full transparency:

### What's Logged:
- **Function Name**: Which function made the AI call
- **Model Used**: Which AI model processed the request
- **Timestamp**: When the call was made
- **Full Prompt**: Complete prompt sent to the AI
- **Full Response**: Complete response received from AI
- **Errors**: Any errors that occurred

### Viewing AI Logs:

**In Real-Time** (console output):
```bash
streamlit run app.py
# Logs appear in terminal as AI calls are made
```

**View Log Files**:
```bash
# View today's log file (Windows)
type app\logs\ai_calls_20251123.log

# View statistics about AI calls
python app/view_ai_logs.py --count

# Show the latest AI call
python app/view_ai_logs.py --latest

# Show all calls to a specific function
python app/view_ai_logs.py --function decide_model
```

### Log File Location:
- **Path**: `app/logs/ai_calls_YYYYMMDD.log`
- **Format**: One file per day
- **Content**: Complete AI interaction history

### Log Analysis Examples:

```bash
# Count AI calls by function
python app/view_ai_logs.py --count
# Output:
# decide_encode_type    : 3 calls
# decide_model         : 2 calls
# decide_fill_null     : 1 call

# View latest AI interaction
python app/view_ai_logs.py --latest
# Shows full prompt and response from last AI call

# Search for specific function calls
python app/view_ai_logs.py --function decide_target_attribute
# Shows all calls to that specific function
```

For more details, see `app/logs/README.md`

## 💰 Cost Comparison

### Before (Original - OpenAI Direct):
- **GPT-4 Turbo**: ~$0.27 per complete analysis
- **GPT-3.5 Turbo**: ~$0.0115 per complete analysis
- **Total Monthly** (100 analyses): $27 - $115

### After (This Fork - OpenRouter Free Models):
- **Grok-4.1-Fast**: $0.00 per analysis ✨
- **GPT-OSS-20B**: $0.00 per analysis ✨
- **Total Monthly** (unlimited analyses): **$0.00**
- **Savings**: **100% cost reduction**

## 🛠️ Troubleshooting

### "API key is required" error?
→ Enter your OpenRouter API key in the password field at the top of the page

### "KeyError: ['mean', '50%', 'std']" error?
→ This bug is fixed in this version. Make sure you're using the latest code.

### Import errors?
→ Run: `pip install -r requirements.txt`

### Model not working?
→ Check your internet connection and verify your API key is valid at [OpenRouter.ai](https://openrouter.ai/)

### Can't find log files?
→ Logs are created in `app/logs/` when you make your first AI call. Directory is auto-created.

### Need more help?
- Check `app/logs/README.md` for logging documentation
- Visit [OpenRouter Documentation](https://openrouter.ai/docs)
- Join [OpenRouter Discord](https://discord.gg/openrouter)

## 📚 Additional Resources

### Model Information:
- **Grok-4.1-Fast**: High-quality analysis, comparable to GPT-4
- **GPT-OSS-20B**: Fast processing, great for quick analyses
- **More models**: Check [OpenRouter Models](https://openrouter.ai/models)

### Documentation:
- **OpenRouter API**: https://openrouter.ai/docs
- **Original Project**: https://github.com/Wilson-ZheLin/Streamline-Analyst
- **Issue Tracking**: https://github.com/dataelvisliang/Streamline-Analyst/issues

## 🎯 Credits & Attribution

This project is built upon the excellent work of **[Zhe Lin](https://github.com/Wilson-ZheLin)** and the original [Streamline Analyst](https://github.com/Wilson-ZheLin/Streamline-Analyst) project.

### Key Modifications in This Fork:

#### ✅ API Migration
- Migrated from OpenAI API to OpenRouter AI
- Removed LangChain dependency for better performance
- Direct API calls using `requests` library

#### ✅ Free Model Implementation
- Added Grok-4.1-Fast (free, high-quality)
- Added GPT-OSS-20B (free, fast processing)
- Model selection in UI with clear "Free" indicators

#### ✅ Security Enhancements
- API key input through secure UI field (password type)
- Keys never stored or saved
- Better error messages with helpful links

#### ✅ Comprehensive Logging System
- All AI calls logged with prompts and responses
- Daily log files for transparency
- Built-in log analysis tools (`view_ai_logs.py`)
- Real-time console logging

#### ✅ Bug Fixes
- Fixed KeyError with non-numeric data columns
- Improved error handling throughout
- Better handling of mixed data types

#### ✅ Documentation & Branding
- Rebranded to AI Analytics Engine
- Enhanced documentation with migration guides
- Clear attribution to original author
- Comprehensive README updates

### Migration Details:
For technical details about the migration, see the migration guides included in the repository or the original migration package documentation.

## 📄 License

This project maintains the same license as the original Streamline Analyst project.

## 🙏 Acknowledgments

- **[Zhe Lin](https://github.com/Wilson-ZheLin)** - Original Streamline Analyst creator
- **OpenRouter** - For providing free tier API access
- **X.AI** - For Grok model access
- **OpenAI** - For GPT-OSS model access
- **Streamlit Community** - For the amazing framework

---

**Ready to start?** Get your free OpenRouter API key and start analyzing data in 5 minutes! 🚀

**Questions?** Check the troubleshooting section or open an issue on GitHub.

**Cost:** $0.00 | **Setup Time:** 5 minutes | **Difficulty:** Easy ✨
