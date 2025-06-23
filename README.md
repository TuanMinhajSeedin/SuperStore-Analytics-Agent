# 🧠 Agentic Application for Analytical Q&A

An intelligent agentic application that can answer analytical questions using natural language processing and data analysis. Built with LangChain, OpenAI, and Streamlit.

## 📋 Overview

This application provides an interactive interface for asking analytical questions about your SuperStore dataset. The agent uses advanced language models to understand queries, retrieve relevant data, perform analysis, and provide insights with visualizations.

### Key Features

- **Natural Language Queries**: Ask questions in plain English
- **Multi-Step Reasoning**: Agent breaks down complex questions into logical steps
- **Data Analysis**: Automatic filtering, aggregation, and statistical analysis
- **Visualizations**: Interactive charts and graphs using Plotly
- **Real-time Insights**: Immediate analysis and response generation
- **Conversation History**: Track your analytical journey
- **Comprehensive Logging**: Detailed logging with file rotation and performance tracking

## 🏗️ Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   LangChain     │    │   OpenAI GPT-4  │
│   Frontend      │◄──►│   Agent         │◄──►│   LLM           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Tools    │    │   Chart Tools   │    │   Vector Store  │
│   • Filtering   │    │   • Bar Charts  │    │   • FAISS       │
│   • Aggregation │    │   • Line Charts │    │   • ChromaDB    │
│   • Statistics  │    │   • Pie Charts  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SuperStore Dataset                           │
│                    (CSV Format)                                │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Logging System                               │
│                    • File Rotation                              │
│                    • Performance Tracking                       │
│                    • Error Monitoring                           │
│                    • Query Analytics                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Agent Workflow

### 1. Query Understanding
- **Input**: Natural language question from user
- **Process**: LLM analyzes intent and breaks down into subtasks
- **Output**: Structured analysis plan
- **Logging**: Query received, analysis type identified

### 2. Data Retrieval
- **Input**: Analysis requirements
- **Process**: Agent selects appropriate data tools
- **Output**: Relevant data subset
- **Logging**: Data operations, performance metrics

### 3. Analysis Execution
- **Input**: Data and analysis plan
- **Process**: Statistical analysis, filtering, aggregation
- **Output**: Quantitative results and insights
- **Logging**: Analysis steps, execution time, results summary

### 4. Response Generation
- **Input**: Analysis results
- **Process**: LLM synthesizes findings into natural language
- **Output**: Comprehensive answer with reasoning
- **Logging**: Response generation, reasoning steps extracted

### 5. Visualization
- **Input**: Analysis results
- **Process**: Automatic chart generation based on data type
- **Output**: Interactive visualizations
- **Logging**: Chart creation, visualization type, data points

## 🚀 How to Run

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TuanMinhajSeedin/SuperStore-Analytics-Agent.git
   cd SuperStore-Analytics-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - Start asking analytical questions!

## 📝 Logging System

The application includes a comprehensive logging system that tracks all operations, performance metrics, and errors.

### Log Features

- **File Rotation**: Automatic log rotation (10MB max, 5 backup files)
- **Multiple Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Structured Format**: Timestamp, component, level, file:line, function, message
- **Performance Tracking**: Operation duration and resource usage
- **Error Monitoring**: Detailed error context and stack traces
- **Query Analytics**: User queries, analysis types, and results

### Log File Location

```
logs/
└── application.logs          # Main log file
├── application.logs.1        # Backup files (rotated)
├── application.logs.2
└── ...
```

### Log Configuration

Configure logging behavior in your `.env` file:

```bash
# Logging Configuration
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR=logs                      # Log directory
LOG_FILE=application.logs         # Log filename
LOG_MAX_SIZE=10485760            # Max file size (10MB)
LOG_BACKUP_COUNT=5               # Number of backup files
LOG_FORMAT=detailed              # "detailed" or "simple"
```

### Log Examples

```
2024-01-15 10:30:15 | analytical_agent | INFO | agent.py:45 | __init__ | Initializing AnalyticalAgent
2024-01-15 10:30:16 | analytical_agent | INFO | agent.py:52 | _load_data | Loading data from file | file_path=artifacts/SuperStoreOrders.csv
2024-01-15 10:30:17 | analytical_agent | INFO | agent.py:58 | _load_data | Data operation: CSV loaded | rows=9994 | columns=21 | file_path=artifacts/SuperStoreOrders.csv
2024-01-15 10:30:18 | streamlit_app | INFO | app.py:45 | main | User query received | query=Which region had the highest sales?
2024-01-15 10:30:19 | analytical_agent | INFO | agent.py:120 | analyze_query | Analysis started | query=Which region had the highest sales? | analysis_type=agent_analysis
2024-01-15 10:30:22 | analytical_agent | INFO | agent.py:140 | analyze_query | Analysis completed | query=Which region had the highest sales? | duration=3.45s | success=true
```

### Monitoring and Debugging

1. **Real-time Monitoring**: Check logs while the application is running
   ```bash
   tail -f logs/application.logs
   ```

2. **Error Analysis**: Search for errors in logs
   ```bash
   grep "ERROR" logs/application.logs
   ```

3. **Performance Analysis**: Find slow operations
   ```bash
   grep "Performance" logs/application.logs
   ```

4. **Query Analytics**: Analyze user behavior
   ```bash
   grep "User query received" logs/application.logs
   ```

## 💡 Example Queries

### Sales Analysis
- "Which region had the highest sales last quarter?"
- "Show me sales trends by month"
- "What are the top 5 products by revenue?"

### Profit Analysis
- "Which category has the highest profit margin?"
- "Show me profit distribution by region"
- "What is the correlation between sales and profit?"

### Performance Metrics
- "Which shipping mode is most profitable?"
- "What is the average order value by category?"
- "Show me customer segment analysis"

## 🛠️ Customization

### Adding New Data Sources

1. **Update data loading in `agent.py`**:
   ```python
   def _load_data(self) -> pd.DataFrame:
       # Add your data loading logic here
       df = pd.read_csv("your_data.csv")
       return df
   ```

2. **Modify data tools in `tools/data_tool.py`**:
   ```python
   def _preprocess_data(self):
       # Add your preprocessing logic
       pass
   ```

### Adding New Chart Types

1. **Extend `tools/chart_tool.py`**:
   ```python
   def _create_custom_chart(self, query: str) -> str:
       # Add your custom chart logic
       pass
   ```

2. **Update the routing logic**:
   ```python
   if 'custom' in query_lower:
       return self._create_custom_chart(query)
   ```

### Modifying Agent Behavior

1. **Update system prompt in `agent.py`**:
   ```python
   self.system_prompt = """Your custom system prompt here"""
   ```

2. **Add new tools**:
   ```python
   self.tools.append(
       Tool(
           name="custom_tool",
           func=self.custom_tool.analyze,
           description="Description of your custom tool"
       )
   )
   ```

### Customizing Logging

1. **Add custom log methods**:
   ```python
   def log_custom_operation(self, operation: str, **kwargs):
       self.logger.info(f"Custom operation: {operation}", **kwargs)
   ```

2. **Modify log format**:
   ```python
   # In utils/logger.py
   detailed_formatter = logging.Formatter(
       '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
   )
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `LANGCHAIN_TRACING_V2` | Enable LangChain tracing | No |
| `LANGCHAIN_ENDPOINT` | LangChain endpoint URL | No |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | No |
| `LOG_DIR` | Log directory path | No |
| `LOG_FILE` | Log filename | No |
| `LOG_MAX_SIZE` | Maximum log file size in bytes | No |
| `LOG_BACKUP_COUNT` | Number of backup log files | No |

### Model Configuration

You can modify the LLM configuration in `agent.py`:

```python
self.llm = ChatOpenAI(
    model="gpt-4",  # Change to gpt-3.5-turbo for cost savings
    temperature=0.1,  # Adjust for creativity vs consistency
    openai_api_key=self.openai_api_key
)
```

## 📊 Data Schema

Data File:  
``https://www.kaggle.com/datasets/laibaanwer/superstore-sales-dataset``

The application expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Order Date | Date | Date of the order |
| Region | String | Geographic region |
| Category | String | Product category |
| Sales | Numeric | Sales amount |
| Profit | Numeric | Profit amount |
| Quantity | Numeric | Quantity ordered |
| Discount | Numeric | Discount percentage |


## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the agentic framework
- [OpenAI](https://openai.com/) for the language models
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for the visualizations
- [Pandas](https://pandas.pydata.org/) for data manipulation

---

**Happy Analyzing! 🎉** 