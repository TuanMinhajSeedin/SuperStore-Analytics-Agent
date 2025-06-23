import pandas as pd
import numpy as np
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import BaseTool, StructuredTool
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import os
import time
from tools.data_tool import DataAnalysisTool
from tools.chart_tool import ChartTool
from config import Config
from utils.logger import get_logger

class AnalyticalAgent:
    """
    An agentic application for analytical Q&A using LangChain and ReAct pattern.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the analytical agent with tools and LLM.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
        """
        # Initialize logger
        self.logger = get_logger("analytical_agent")
        self.logger.info("Initializing AnalyticalAgent")
        
        try:
            # Validate configuration
            Config.validate()
            self.logger.info("Configuration validation passed")
            
            self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
            
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to the constructor.")
            
            # Initialize LLM
            openai_config = Config.get_openai_config()
            self.llm = ChatOpenAI(
                model=openai_config["model"],
                temperature=openai_config["temperature"],
                openai_api_key=self.openai_api_key
            )
            self.logger.info("LLM initialized", model=openai_config["model"], temperature=openai_config["temperature"])
            
            # Load data
            self.data_path = Config.DATA_PATH
            self.df = self._load_data()
            self.logger.log_data_operation("Data loaded", self.df.shape, file_path=self.data_path)
            
            # Initialize tools
            self.data_tool = DataAnalysisTool(self.df)
            self.chart_tool = ChartTool(self.df)
            self.logger.info("Analysis tools initialized")
            
            # Create tools for LangChain
            self.tools = [
                StructuredTool.from_function(
                    func=self.data_tool.analyze,
                    name="data_analysis",
                    description="Use this tool to analyze data with various operations. Input should be a natural language query describing what you want to analyze. Examples: 'Show me sales by region', 'What is the total profit by category?', 'Filter data for the last quarter', 'Calculate average sales by month'"
                ),
                StructuredTool.from_function(
                    func=self.chart_tool.create_chart,
                    name="create_chart",
                    description="Use this tool to create visualizations and charts from data. Input should be a natural language query describing what type of chart you want. Examples: 'Create a bar chart of sales by region', 'Show me a pie chart of sales by category', 'Create a line chart showing sales trends over time'"
                )
            ]
            self.logger.info("LangChain tools created", tool_count=len(self.tools))
            
            # Initialize agent
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15,
                early_stopping_method="generate",
                return_intermediate_steps=True
            )
            self.logger.info("LangChain agent initialized successfully")
            
            # System prompt for analytical reasoning
            self.system_prompt = """You are an expert data analyst and business intelligence specialist. Your role is to:

1. UNDERSTAND the user's analytical question and break it down into clear steps
2. USE the data_analysis tool to retrieve and analyze relevant data
3. USE the create_chart tool to create visualizations when needed
4. PROVIDE clear, actionable insights with proper context

IMPORTANT INSTRUCTIONS:
- ALWAYS start by using the data_analysis tool to get the data you need
- Use the create_chart tool to create visualizations that will help explain your findings
- Be specific and detailed in your analysis
- Provide numerical results with proper formatting
- Explain your findings clearly and provide business insights
- DO NOT repeat the same analysis multiple times
- If a tool doesn't give the expected result, try a different approach or provide the analysis manually

Available data columns: {columns}

WORKFLOW:
1. First, use data_analysis tool with a clear query about what data you need
2. Analyze the results from the data_analysis tool
3. Use create_chart tool to create a relevant visualization
4. Provide a comprehensive summary with insights and recommendations

Example queries for data_analysis tool:
- "Show me sales by region"
- "What is the total profit by category?"
- "Calculate average sales by month"
- "Filter data for the last quarter"
- "Calculate profit margin by region"
- "What are the top 10 products by sales?"

Example queries for create_chart tool:
- "Create a bar chart of sales by region"
- "Show me a pie chart of sales by category"
- "Create a line chart showing sales trends over time"
- "Create a bar chart of top 10 products by sales"

Remember to:
- Be thorough but concise
- Explain your methodology
- Provide both quantitative and qualitative insights
- Format numbers appropriately (currency, percentages, etc.)
- Always use the tools to get actual data and create visualizations
- Avoid repetitive loops - if a tool doesn't work as expected, move on to the next step
"""

            self.logger.info("AnalyticalAgent initialization completed successfully")
            
        except Exception as e:
            self.logger.log_error(e, "AnalyticalAgent initialization")
            raise

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the SuperStore data."""
        start_time = time.time()
        self.logger.info("Loading data from file", file_path=self.data_path)
        
        try:
            df = pd.read_csv(self.data_path)
            self.logger.log_data_operation("CSV loaded", df.shape, file_path=self.data_path)
            
            # Convert date columns with proper format
            if 'order_date' in df.columns:
                # Handle the M/D/YYYY format
                df['order_date'] = pd.to_datetime(df['order_date'], format='%m/%d/%Y', errors='coerce')
                df['year'] = df['order_date'].dt.year
                df['month'] = df['order_date'].dt.month
                df['quarter'] = df['order_date'].dt.quarter
                self.logger.info("Date columns processed")
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['sales', 'profit', 'quantity', 'discount']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    self.logger.debug(f"Converted column to numeric", column=col)
            
            duration = time.time() - start_time
            self.logger.log_performance("Data loading and preprocessing", duration, rows=len(df), columns=len(df.columns))
            
            return df
            
        except Exception as e:
            self.logger.log_error(e, "Data loading", file_path=self.data_path)
            raise Exception(f"Error loading data: {e}")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query and return a comprehensive response.
        
        Args:
            query: The user's analytical question
            
        Returns:
            Dictionary containing answer, reasoning steps, data insights, and visualization
        """
        start_time = time.time()
        self.logger.log_query(query)
        self.logger.log_analysis_start(query, "agent_analysis")
        
        try:
            # Update system prompt with current data columns
            system_prompt = self.system_prompt.format(
                columns=", ".join(self.df.columns.tolist())
            )
            
            # Create the full prompt
            full_prompt = f"""
{system_prompt}

User Question: {query}

Please analyze this question step by step:

1. First, understand what the user is asking for
2. Determine what data and analysis is needed
3. Use the appropriate tools to retrieve and analyze the data
4. Provide a clear, comprehensive answer with insights
5. Create a visualization if it would be helpful

Remember to:
- Show your reasoning process
- Provide specific numbers and insights
- Explain the business implications
- Suggest follow-up questions if relevant
"""

            self.logger.debug("Sending query to agent", query_length=len(query))
            
            # Get agent response using invoke instead of run
            response = self.agent.invoke({"input": full_prompt})
            agent_response = response.get("output", str(response))
            self.logger.debug("Agent response received", response_length=len(agent_response))
            
            # Extract visualization from intermediate steps
            visualization = None
            if "intermediate_steps" in response:
                for action, tool_output in response["intermediate_steps"]:
                    if action.tool == "create_chart" and isinstance(tool_output, go.Figure):
                        visualization = tool_output
                        self.logger.info("Visualization found in agent's intermediate steps.")
                        break

            # Parse the response to extract different components
            parsed_response = self._parse_agent_response(agent_response, query, visualization)
            
            duration = time.time() - start_time
            self.logger.log_analysis_complete(query, duration, success=True)
            
            return parsed_response
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_analysis_complete(query, duration, success=False)
            self.logger.log_error(e, "Query analysis", query=query)
            
            # Fallback to direct analysis if agent fails
            return self._fallback_analysis(query, str(e))

    def _parse_agent_response(self, response: str, original_query: str, visualization: Optional[go.Figure] = None) -> Dict[str, Any]:
        """
        Parse the agent's response to extract different components.
        
        Args:
            response: Raw response from the agent
            original_query: Original user query
            visualization: An optional visualization created by the agent.
            
        Returns:
            Structured response dictionary
        """
        self.logger.debug("Parsing agent response", response_length=len(response))
        
        try:
            # Extract reasoning steps (look for numbered steps or bullet points)
            reasoning_steps = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line) or line.startswith('•') or line.startswith('-'):
                    reasoning_steps.append(line)
            
            self.logger.debug("Extracted reasoning steps", step_count=len(reasoning_steps))
            
            # Extract key insights using the LLM
            data_insights = self._extract_insights(response)
            
            return {
                'answer': response,
                'reasoning_steps': reasoning_steps,
                'data_insights': data_insights,
                'visualization': visualization
            }
            
        except Exception as e:
            self.logger.log_error(e, "Response parsing", query=original_query)
            return {
                'answer': response,
                'reasoning_steps': [],
                'data_insights': "Error parsing response",
                'visualization': visualization
            }

    def _extract_insights(self, response: str) -> str:
        """
        Extract key insights from the response using an LLM.
        
        Args:
            response: Agent response text
            
        Returns:
            Formatted insights string
        """
        self.logger.debug("Extracting insights from response with LLM", response_length=len(response))
        
        try:
            insight_prompt = (
                "Based on the following data analysis response, please extract and summarize the key insights into a few bullet points.\n"
                "Focus on the most important findings, trends, or conclusions.\n\n"
                "Analysis Response:\n"
                "---\n"
                f"{response}\n"
                "---\n\n"
                "Key Insights (as bullet points):"
            )
            insight_response = self.llm.invoke([HumanMessage(content=insight_prompt)])
            insights = insight_response.content.strip()
            
            self.logger.debug("LLM-extracted insights", insights=insights)
            return insights if insights else "Analysis completed successfully."

        except Exception as e:
            self.logger.log_error(e, "Insight extraction with LLM")
            # Fallback to the old method if LLM fails
            return self._extract_insights_rule_based(response)

    def _extract_insights_rule_based(self, response: str) -> str:
        """
        Extract key insights from the response.
        
        Args:
            response: Agent response text
            
        Returns:
            Formatted insights string
        """
        self.logger.debug("Extracting insights from response", response_length=len(response))
        
        try:
            # Look for key metrics and insights in the response
            insights = []
            
            # Extract numbers and percentages
            numbers = re.findall(r'\$[\d,]+\.?\d*', response)
            percentages = re.findall(r'\d+\.?\d*%', response)
            
            if numbers:
                insights.append(f"Key financial metrics: {', '.join(numbers[:3])}")
            if percentages:
                insights.append(f"Key percentages: {', '.join(percentages[:3])}")
                
            # Look for comparative statements
            comparative_words = ['higher', 'lower', 'best', 'worst', 'top', 'bottom']
            for word in comparative_words:
                if word in response.lower():
                    # Find the sentence containing this word
                    sentences = response.split('.')
                    for sentence in sentences:
                        if word in sentence.lower():
                            insights.append(sentence.strip())
                            break
                    break
            
            result = ' '.join(insights) if insights else "Analysis completed successfully."
            self.logger.debug("Insights extracted", insight_count=len(insights))
            return result
            
        except Exception as e:
            self.logger.log_error(e, "Insight extraction")
            return "Error extracting insights."

    def _fallback_analysis(self, query: str, error_msg: str) -> Dict[str, Any]:
        """
        Fallback analysis when the agent fails.
        
        Args:
            query: Original user query
            error_msg: Error message from the agent
            
        Returns:
            Basic analysis response
        """
        self.logger.warning("Using fallback analysis", query=query, error=error_msg)
        
        answer = f"I apologize, but I encountered an error while processing your query. Please try rephrasing your question or contact support.\n\n**Error details:** {error_msg}"
        
        return {
            'answer': answer,
            'reasoning_steps': ["Agent execution failed. Used fallback response."],
            'data_insights': "No insights generated due to an error.",
            'visualization': None
        } 