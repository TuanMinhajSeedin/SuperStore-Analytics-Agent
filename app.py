import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from agent import AnalyticalAgent
from config import Config
from utils.logger import get_logger

# Initialize logger
logger = get_logger("streamlit_app")
load_dotenv()

def ensure_directories():
    """Ensure required directories exist."""
    try:
        # Ensure logs directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        logger.info("Logs directory ensured")
        
        # Ensure artifacts directory exists
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        logger.info("Artifacts directory ensured")
        
    except Exception as e:
        logger.log_error(e, "Directory creation")
        st.error(f"Error creating required directories: {e}")

def load_quick_insights():
    """Load predefined quick insights."""
    try:
        insights = [
            {
                "title": "Top Performing Regions",
                "query": "Which regions have the highest sales?",
                "description": "Discover the best-performing regions by sales volume"
            },
            {
                "title": "Profit Analysis",
                "query": "What is the profit margin by category?",
                "description": "Analyze profitability across different product categories"
            },
            {
                "title": "Sales Trends",
                "query": "Show me sales trends over time",
                "description": "Visualize how sales have changed over the months"
            },
            {
                "title": "Product Performance",
                "query": "What are the top 10 products by sales?",
                "description": "Identify the best-selling products in the dataset"
            },
            {
                "title": "Regional Comparison",
                "query": "Compare sales and profit by region",
                "description": "See how different regions perform in sales vs profit"
            },
            {
                "title": "Discount Impact",
                "query": "How do discounts affect sales and profit?",
                "description": "Analyze the relationship between discounts and performance"
            }
        ]
        logger.info("Quick insights loaded successfully", insight_count=len(insights))
        return insights
    except Exception as e:
        logger.log_error(e, "Loading quick insights")
        return []

def main():
    """Main Streamlit application."""
    start_time = datetime.now()
    logger.info("Starting Streamlit application")
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Page configuration
        st.set_page_config(
            page_title="SuperStore Analytics Agent",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .insight-card {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .insight-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">SuperStore Analytics Agent</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Ask questions about your data and get intelligent insights with AI-powered analysis</p>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            logger.info("Initialized conversation history")
        
        if 'agent' not in st.session_state:
            try:
                # Initialize the agent
                # api_key = st.secrets.get("OPENAI_API_KEY", None)
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                
                if not api_key:
                    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit secrets.")
                    logger.error("OpenAI API key not found")
                    return
                
                st.session_state.agent = AnalyticalAgent(openai_api_key=api_key)
                logger.info("Analytical agent initialized successfully")
                
            except Exception as e:
                st.error(f"❌ Error initializing agent: {str(e)}")
                logger.log_error(e, "Agent initialization")
                return
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Model settings
            st.subheader("Model Settings")
            temperature = st.slider("Temperature", value=0.1, min_value=0.0, max_value=1.0, step=0.1, help="Controls randomness in responses")
            
            # Data overview
            st.subheader("Data Overview")
            try:
                agent = st.session_state.agent
                total_records = len(agent.df)
                total_sales = agent.df['sales'].sum() if 'sales' in agent.df.columns else 0
                total_profit = agent.df['profit'].sum() if 'profit' in agent.df.columns else 0
                
                st.metric("Total Records", f"{total_records:,}")
                st.metric("Total Sales", f"${total_sales:,.0f}")
                st.metric("Total Profit", f"${total_profit:,.0f}")
                
                logger.info("Data overview displayed", 
                           total_records=total_records, 
                           total_sales=total_sales, 
                           total_profit=total_profit)
                
            except Exception as e:
                st.error(f"Error loading data overview: {e}")
                logger.log_error(e, "Data overview display")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            if st.button("Clear History", help="Clear conversation history"):
                st.session_state.conversation_history = []
                st.rerun()
                logger.info("Conversation history cleared")
            
            if st.button("Export Logs", help="Download application logs"):
                try:
                    log_file = Path("logs/application.logs")
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            log_content = f.read()
                        st.download_button(
                            label="Download Logs",
                            data=log_content,
                            file_name=f"analytics_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        logger.info("Log export initiated")
                    else:
                        st.warning("No log file found")
                        logger.warning("Log file not found for export")
                except Exception as e:
                    st.error(f"Error exporting logs: {e}")
                    logger.log_error(e, "Log export")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Ask Your Question")
            
            # Query input
            query = st.text_area(
                "Enter your analytical question:",
                placeholder="e.g., Which region had the highest sales last quarter?",
                height=100,
                help="Ask any question about your SuperStore data"
            )
            
            # Analysis button
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                if query.strip():
                    logger.log_query(query)
                    
                    with st.spinner("Analyzing your question..."):
                        try:
                            # Get agent response
                            response = st.session_state.agent.analyze_query(query)
                            # st.write("response", response)
                            logger.info("Analysis completed successfully", query=query)
                            
                            # Store in conversation history
                            st.session_state.conversation_history.append({
                                'timestamp': datetime.now(),
                                'query': query,
                                'response': response
                            })
                            logger.info("Response stored in conversation history", history_length=len(st.session_state.conversation_history))
                            
                            # Display results
                            st.success("✅ Analysis complete!")
                            
                            # Answer
                            st.subheader("Answer")
                            st.write(response['answer'])
                            
                            # Reasoning steps
                            if response.get('reasoning_steps'):
                                st.subheader("🧠 Reasoning Steps")
                                for i, step in enumerate(response['reasoning_steps'], 1):
                                    st.write(f"{i}. {step}")
                            
                            # Data insights
                            if response.get('data_insights'):
                                st.subheader("💡 Key Insights")
                                st.info(response['data_insights'])
                            
                            # Visualization
                            if response.get('visualization'):
                                st.subheader("Visualization")
                                st.plotly_chart(response['visualization'], use_container_width=True)
                            
                            logger.info("Response displayed successfully")
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            logger.log_error(e, "Query analysis", query=query)
                else:
                    st.warning("⚠️ Please enter a question to analyze.")
        
        with col2:
            st.header("Quick Insights")
            
            try:
                insights = load_quick_insights()
                
                for insight in insights:
                    with st.container():
                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>{insight['title']}</h4>
                            <p>{insight['query']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # if st.button(f"Ask: {insight['title']}", key=insight['title']):
                        #     st.session_state.quick_query = insight['query']
                        #     st.rerun()
                
                # logger.info("Quick insights displayed", insight_count=len(insights))
                
            except Exception as e:
                st.error(f"Error loading quick insights: {e}")
                logger.log_error(e, "Loading quick insights")

            # with st.expander("Quick Insights"):
            #     insights = load_quick_insights()
            #     for insight in insights:
            #         st.markdown(f"""
            #         <div class="insight-card">
            #             <h4>{insight['title']}</h4>
            #             <p>{insight['query']}</p>
            #         </div>
            #         """, unsafe_allow_html=True)
        
        # Handle quick query from sidebar
        if 'quick_query' in st.session_state:
            query = st.session_state.quick_query
            del st.session_state.quick_query
            st.rerun()
        
        # Conversation history
        if st.session_state.conversation_history:
            st.header("Conversation History")
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q: {conv['query'][:50]}... ({conv['timestamp'].strftime('%H:%M:%S')})"):
                    st.write(f"**Question:** {conv['query']}")
                    st.write(f"**Answer:** {conv['response']['answer']}")
                    
                    if conv['response'].get('reasoning_steps'):
                        st.write("**Reasoning Steps:**")
                        for j, step in enumerate(conv['response']['reasoning_steps'], 1):
                            st.write(f"{j}. {step}")
                    
                    if conv['response'].get('data_insights'):
                        st.info(f"**Insights:** {conv['response']['data_insights']}")
        
        duration = datetime.now() - start_time
        logger.info("Streamlit application completed successfully", duration=duration.total_seconds())
        
    except Exception as e:
        logger.log_error(e, "Streamlit application")
        st.error(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    main() 