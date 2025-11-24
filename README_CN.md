# AI Analytics Engine: 数据分析领域的 AI Agent

**基于 [Zhe Lin](https://github.com/Wilson-ZheLin) 开发的 [Streamline Analyst](https://github.com/Wilson-ZheLin/Streamline-Analyst) 项目构建**

---

**AI Analytics Engine**🪄是一个开源的基于大语言模型（LLM）的应用，目标简化数据分析中从数据清洗到模型测试的全部流程。分类预测、聚类、回归、数据集可视化、数据预处理、编码、特征选择、目标属性判断、可视化、最佳模型选择等等任务都可自主决策和执行。用户需要做的只有**选择数据文件**、**选择分析模式**，剩下的工作就可以让AI来接管了🔮。所有处理后的数据和训练的模型都可下载。

**由 OpenRouter AI 提供支持** - 免费访问多个AI模型，包括 Grok-4.1 和 GPT-OSS，完全免费！

*所有上传的数据和API Keys不会以任何形式储存或分享！

![Screenshot 2024-02-12 at 16 01 01](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/4167b04c-0853-4703-87a4-6c2994e30f9e)

未来版本预期更新功能：***自然语言处理 (NLP)***、***卷积/循环神经网络***、***目标检测 (基于YOLO)***...

## 此分支的新功能

- **🆓 免费AI模型**: 从 OpenAI API 迁移到 OpenRouter AI，提供 Grok-4.1-Fast 和 GPT-OSS-20B 等免费模型访问
- **🔍 全面日志记录**: 所有AI调用都会被记录，包括完整的提示词和响应，便于透明度和调试
- **🎨 全新品牌界面**: 更新为 AI Analytics Engine 品牌，同时保留所有原有功能
- **📊 增强分析工具**: 为AI交互添加了详细的日志记录和分析工具

主页
----

https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/1d30faca-f474-42fd-b20b-c93ed7cf6d13

当前版本功能
----------

* **目标变量识别**: 若LLM无法确定，则提醒用户选择
* **空值管理**: 由LLM根据每列数据信息从均值、中位数、众数填充、插值，或引入新类别等策略中选择
* **数据编码**: 根据每列数据信息判断使用：独热编码、整数映射或标签编码
* **PCA降维**
* **处理重复实体**
* **数据转换和标准化**: 利用 Box-Cox 转换和标准化优化数据分布和可扩展性
* **平衡目标变量实体**: LLM 推荐的方法如随机过采样、SMOTE 和 ADASYN 帮助平衡数据集，对于无偏见模型训练至关重要
* **数据集划分比例**: LLM 确定数据集的比例（也可以手动调整）
* **模型选择和训练**: LLM 根据数据推荐并使用最适合的模型进行训练
* **群集数量推荐**: 对于聚类任务，使用肘部法则和轮廓系数推荐最佳群集数量（可手动调整）

- 所有处理过的数据和模型都可供下载

### 建模和结果可视化:

![Screenshot 2024-02-12 at 16 10 35](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/423da7be-63f1-491d-9ebe-6a788c440c40)

### 自动化工作流界面:

![Screenshot 2024-02-12 at 16 20 19](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/9d04d5f2-4f2a-44eb-ab8b-c07c8c0c5a53)

### 支持的建模任务：

| **分类模型**                      | **聚类模型**                   | **回归模型**                         |
|----------------------------------|-------------------------------|-------------------------------------|
| 逻辑回归                          | K-均值聚类                    | 线性回归                             |
| 随机森林                          | DBSCAN                        | 岭回归                               |
| 支持向量机                        | 高斯混合模型                  | Lasso回归                            |
| 梯度提升机                        | 层次聚类                      | 弹性网回归                           |
| 高斯朴素贝叶斯                    | 谱聚类                        | 随机森林回归                         |
| AdaBoost                          | 其他                          | 梯度提升回归                         |
| XGBoost                           |                               | 其他                                 |

### 实时计算模型指标与结果可视化：

| **分类指标 & 图表**                | **聚类指标 & 图表**            | **回归指标 & 图表**                   |
|------------------------------------|--------------------------------|---------------------------------------|
| 模型分数                            | 轮廓分数                        | R平方分数                             |
| 混淆矩阵                            | Calinski-Harabasz 分数         | 均方误差 (MSE)                        |
| AUC                                 | Davies-Bouldin 分数            | 均方根误差 (RMSE)                     |
| F1 分数                             | 聚类散点图                      | 绝对误差 (MAE)                        |
| ROC 曲线                            | 其他                           | 残差图                                |
| 其他                                |                                | 预测值 vs 实际值图                    |
|                                    |                                | 分位数-分位数图                       |


### 可视化分析工具包:

AI Analytics Engine 🪄 提供了一系列直观的可视化工具，这部分的使用**无需 API Key**：

* **单属性可视化**: 深入个别数据方面的洞察视图
* **多属性可视化**: 变量间关系的全面分析
* **三维绘图**: 复杂数据关系的3D可视化
* **词云**: 通过词频突出关键主题和概念
* **世界热力图**: 使地理趋势和分布可视化
* 更多图表正在开发中...


本地运行安装
----------

### 环境&前置准备

运行 `app.py`, 首先需要:
* [Python 3.11.5](https://www.python.org/downloads/)
* [OpenRouter API Key](https://openrouter.ai/) - **完全免费使用！**
    * 在 [OpenRouter.ai](https://openrouter.ai/) 获取免费API密钥
    * Grok-4.1-Fast 和 GPT-OSS-20B 模型完全免费
    * 无需信用卡

### 安装和运行

1. 克隆此仓库
```bash
git clone https://github.com/dataelvisliang/Streamline-Analyst.git
cd Streamline-Analyst
```

2. 安装所需的依赖包
```bash
pip install -r requirements.txt
```

3. 运行应用程序
```bash
cd app
streamlit run app.py
```

4. 在网页界面中输入您的 OpenRouter API 密钥，然后开始分析！

### 查看AI调用日志

所有AI交互都会自动记录。要查看和分析日志：

```bash
# 查看AI调用的统计信息
python app/view_ai_logs.py --count

# 显示最新的AI调用
python app/view_ai_logs.py --latest

# 显示特定函数的所有调用
python app/view_ai_logs.py --function decide_model
```

日志文件存储在 `app/logs/ai_calls_YYYYMMDD.log`

## 致谢

本项目基于 **[Zhe Lin](https://github.com/Wilson-ZheLin)** 的优秀工作和原始 [Streamline Analyst](https://github.com/Wilson-ZheLin/Streamline-Analyst) 项目构建。

### 主要修改：
- 从 OpenAI API 迁移到 OpenRouter AI 以实现免费模型访问
- 为所有AI交互添加了全面的日志系统
- 重新品牌为 AI Analytics Engine
- 增强了文档和分析工具

## 许可证

本项目保持与原始 Streamline Analyst 项目相同的许可证。
