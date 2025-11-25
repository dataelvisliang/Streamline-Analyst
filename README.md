# AI Analytics Engine: A Data Analysis AI Agent

**Built on [Streamline Analyst](https://github.com/Wilson-ZheLin/Streamline-Analyst) by [Zhe Lin](https://github.com/Wilson-ZheLin)**

AI Analytics Engine is an open-source, LLM-powered application that automates data analysis workflows. Just **upload your data**, **ask questions** or **select an analysis mode**, and **start analyzing**.

**Powered by OpenRouter AI** - Access free AI models including Grok-4.1-Fast and GPT-OSS-20B!

![AI Analytics Engine](https://github.com/dataelvisliang/Streamline-Analyst/blob/main/assets/AI%20Analytics%20Engine.png)

### Key Features
- **AI Planning Assistant**: Chat with AI to determine the best analysis approach
- **Automated Workflows**: Data cleaning, preprocessing, model selection, and training
- **100% Free Models**: Grok-4.1-Fast and GPT-OSS-20B at zero cost
- **Context-Aware Results**: AI summaries relate findings to your original questions
- **Smart Fallbacks**: Helpful suggestions when analysis can't proceed
- **Privacy First**: Data and API keys never stored or shared

## 🎉 What's New

### Latest Updates (v2.1.0 - Nov 2024)
- **💬 AI Planning Assistant**: Chat interface to help determine the best analysis approach
- **🤖 Autonomous Execution**: Planning agent automatically starts analysis when ready
- **📊 Context-Aware Summaries**: AI relates findings to your original questions
- **🔄 Smart Fallbacks**: Suggests alternatives when data doesn't fit selected mode
- **📥 Better UX**: Download buttons repositioned, cleaner interface

### Version 2.0.0 Features
- **🆓 100% Free AI Models**: Grok-4.1-Fast and GPT-OSS-20B
- **🔐 Secure API Keys**: UI-based entry, never stored
- **🔍 AI Logging**: Full transparency with daily log files
- **⚡ 90% Faster**: Removed LangChain dependency
- **🐛 Bug Fixes**: KeyError fixes, better error handling

## 📺 Demo

https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/1d30faca-f474-42fd-b20b-c93ed7cf6d13

## ✨ Core Capabilities

* **AI Planning Assistant**: Natural language interface to determine analysis needs
* **Target Variable Identification**: LLM identifies the target variable
* **Missing Value Management**: AI-recommended strategies (mean, median, mode, interpolation, new categories)
* **Smart Encoding**: Automated one-hot, integer mapping, and label encoding
* **PCA Dimensionality Reduction**: Automatic feature reduction when needed
* **Data Transformation**: Box-Cox transformation and normalization
* **Class Balancing**: SMOTE, ADASYN, random over-sampling (LLM-recommended)
* **Model Selection**: AI recommends and trains best-fit models
* **Cluster Optimization**: Elbow Rule and Silhouette Coefficient for optimal clusters
* **Context-Aware Results**: AI summaries relate to your original questions
* **Full Transparency**: All AI interactions logged for debugging

## 🤖 Supported Models

| **Classification Models**        | **Clustering Models**         | **Regression Models**               |
|----------------------------------|-------------------------------|-------------------------------------|
| Logistic regression              | K-means clustering            | Linear regression                   |
| Random forest                    | DBSCAN                        | Ridge regression                    |
| Support vector machine           | Gaussian mixture model        | Lasso regression                    |
| Gradient boosting machine        | Hierarchical clustering       | Elastic net regression              |
| Gaussian Naive Bayes             | Spectral clustering           | Random forest regression            |
| AdaBoost                         | etc.                          | Gradient boosting regression        |
| XGBoost                          |                               | etc.                                |

## 📊 Metrics & Visualizations

| **Classification** | **Clustering** | **Regression** |
|-------------------|----------------|----------------|
| Accuracy, F1, AUC | Silhouette score | R², MSE, RMSE, MAE |
| Confusion matrix | Calinski-Harabasz | Residual plots |
| ROC curves | Davies-Bouldin | Predicted vs Actual |

### Visual Analysis (No API Key Required)
* Single & multi-attribute visualizations
* 3D plotting
* Word clouds
* World heat maps

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/dataelvisliang/Streamline-Analyst.git
cd Streamline-Analyst
pip install -r requirements.txt

# 2. Run
cd app
streamlit run app.py

# 3. Get free API key at https://openrouter.ai/
# 4. Upload data and start analyzing!
```

**Requirements**: Python 3.11.5+ | **Cost**: $0.00 | **Time**: 5 minutes

## 🔍 AI Transparency

All AI interactions are logged with full transparency in `app/logs/ai_calls_YYYYMMDD.log`.

View logs:
```bash
# View statistics
python app/view_ai_logs.py --count

# Show latest AI call
python app/view_ai_logs.py --latest
```

See `app/logs/README.md` for details.

## 📚 Resources

- **OpenRouter API**: https://openrouter.ai/docs
- **Original Project**: https://github.com/Wilson-ZheLin/Streamline-Analyst
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)
- **Issues**: https://github.com/dataelvisliang/Streamline-Analyst/issues

## 🙏 Acknowledgments

- **[Zhe Lin](https://github.com/Wilson-ZheLin)** - Original Streamline Analyst creator
- **OpenRouter** - Free tier API access
- **Streamlit** - Amazing framework

---

**Ready to start?** Get your free API key at [OpenRouter.ai](https://openrouter.ai/) and start analyzing! 🚀
