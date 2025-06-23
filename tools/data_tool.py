import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime, timedelta
import time
from utils.logger import get_logger

class DataAnalysisTool:
    """
    Tool for performing various data analysis operations on the SuperStore dataset.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data analysis tool with a DataFrame.
        
        Args:
            df: Pandas DataFrame containing the data to analyze
        """
        self.df = df.copy()
        self.logger = get_logger("data_analysis_tool")
        self.logger.info("Initializing DataAnalysisTool", data_shape=self.df.shape)
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the data for better analysis."""
        start_time = time.time()
        self.logger.info("Starting data preprocessing")
        
        try:
            # Convert date columns with proper format
            if 'order_date' in self.df.columns:
                # Handle the M/D/YYYY format
                self.df['order_date'] = pd.to_datetime(self.df['order_date'], format='%m/%d/%Y', errors='coerce')
                self.df['year'] = self.df['order_date'].dt.year
                self.df['month'] = self.df['order_date'].dt.month
                self.df['quarter'] = self.df['order_date'].dt.quarter
                self.df['month_name'] = self.df['order_date'].dt.strftime('%B')
                self.logger.info("Date columns processed")
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['sales', 'profit', 'quantity', 'discount']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    self.logger.debug(f"Converted column to numeric", column=col)
            
            # Calculate additional metrics
            if 'sales' in self.df.columns and 'profit' in self.df.columns:
                self.df['profit_margin'] = (self.df['profit'] / self.df['sales']) * 100
                self.df['profit_margin'] = self.df['profit_margin'].fillna(0)
                self.logger.info("Profit margin calculated")
            
            duration = time.time() - start_time
            self.logger.log_performance("Data preprocessing", duration, data_shape=self.df.shape)
            
        except Exception as e:
            self.logger.log_error(e, "Data preprocessing")
            raise
    
    def analyze(self, query: str) -> str:
        """
        Main analysis method that interprets the query and performs appropriate analysis.
        
        Args:
            query: Analysis query or instruction
            
        Returns:
            Analysis results as a formatted string
        """
        start_time = time.time()
        self.logger.info("Starting data analysis", query=query)
        
        try:
            query_lower = query.lower()
            
            # Route to appropriate analysis method based on query content
            if any(word in query_lower for word in ['filter', 'where', 'condition']):
                result = self._filter_data(query)
            elif any(word in query_lower for word in ['group', 'by', 'aggregate']):
                result = self._group_and_aggregate(query)
            elif any(word in query_lower for word in ['statistic', 'summary', 'describe']):
                result = self._statistical_summary(query)
            elif any(word in query_lower for word in ['trend', 'time', 'period']):
                result = self._time_series_analysis(query)
            elif any(word in query_lower for word in ['correlation', 'relationship']):
                result = self._correlation_analysis(query)
            elif any(word in query_lower for word in ['top', 'bottom', 'rank']):
                result = self._ranking_analysis(query)
            elif any(word in query_lower for word in ['profit margin', 'margin']):
                result = self._profit_margin_analysis(query)
            else:
                result = self._general_analysis(query)
            
            duration = time.time() - start_time
            self.logger.log_performance("Data analysis", duration, query=query, result_length=len(result))
            
            return result
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, "Data analysis", query=query, duration=duration)
            return f"Error in analysis: {str(e)}"
    
    def _filter_data(self, query: str) -> str:
        """Filter data based on conditions in the query."""
        self.logger.debug("Applying data filtering", query=query)
        
        try:
            # Extract filter conditions from query
            conditions = []
            
            # Date filters
            if 'last quarter' in query.lower():
                current_date = datetime.now()
                last_quarter_start = current_date - timedelta(days=90)
                conditions.append(f"order_date >= '{last_quarter_start.strftime('%Y-%m-%d')}'")
                self.logger.debug("Applied last quarter filter")
            
            if 'last month' in query.lower():
                current_date = datetime.now()
                last_month_start = current_date - timedelta(days=30)
                conditions.append(f"order_date >= '{last_month_start.strftime('%Y-%m-%d')}'")
                self.logger.debug("Applied last month filter")
            
            if 'this year' in query.lower():
                current_year = datetime.now().year
                conditions.append(f"year == {current_year}")
                self.logger.debug("Applied this year filter")
            
            # Region filters
            region_match = re.search(r'region\s+(?:is\s+)?([a-zA-Z\s]+)', query, re.IGNORECASE)
            if region_match:
                region = region_match.group(1).strip()
                conditions.append(f"region == '{region}'")
                self.logger.debug("Applied region filter", region=region)
            
            # Category filters
            category_match = re.search(r'category\s+(?:is\s+)?([a-zA-Z\s]+)', query, re.IGNORECASE)
            if category_match:
                category = category_match.group(1).strip()
                conditions.append(f"category == '{category}'")
                self.logger.debug("Applied category filter", category=category)
            
            # Apply filters
            filtered_df = self.df.copy()
            for condition in conditions:
                try:
                    filtered_df = filtered_df.query(condition)
                    self.logger.debug("Applied filter condition", condition=condition)
                except:
                    self.logger.warning("Failed to apply filter condition", condition=condition)
                    continue
            
            # Return summary of filtered data
            total_sales = filtered_df['sales'].sum() if 'sales' in filtered_df.columns else 0
            total_profit = filtered_df['profit'].sum() if 'profit' in filtered_df.columns else 0
            record_count = len(filtered_df)
            
            self.logger.info("Data filtering completed", 
                           original_count=len(self.df), 
                           filtered_count=record_count,
                           conditions_applied=len(conditions))
            
            return f"Filtered data contains {record_count:,} records with total sales of ${total_sales:,.2f} and total profit of ${total_profit:,.2f}."
            
        except Exception as e:
            self.logger.log_error(e, "Data filtering", query=query)
            return f"Error in data filtering: {str(e)}"
    
    def _group_and_aggregate(self, query: str) -> str:
        """Group data and perform aggregations."""
        self.logger.debug("Performing group and aggregate operation", query=query)
        
        try:
            # Determine grouping column
            group_by = None
            query_lower = query.lower()
            if 'product_name' in query_lower and 'product_name' in self.df.columns:
                group_by = 'product_name'
            elif 'product' in query_lower or 'item' in query_lower:
                group_by = 'product_name' if 'product_name' in self.df.columns else 'category'
            elif 'region' in query_lower:
                group_by = 'region'
            elif 'category' in query_lower:
                group_by = 'category'
            elif 'month' in query_lower:
                group_by = 'month_name'
            elif 'quarter' in query_lower:
                group_by = 'quarter'
            elif 'year' in query_lower:
                group_by = 'year'
            
            if not group_by or group_by not in self.df.columns:
                self.logger.warning("Unable to determine grouping column", query=query)
                return "Unable to determine grouping column from query."
            
            # Determine aggregation metric
            metric = 'sales'
            if 'profit' in query_lower:
                metric = 'profit'
            elif 'quantity' in query_lower:
                metric = 'quantity'
            
            if metric not in self.df.columns:
                self.logger.warning("Metric not found in dataset", metric=metric)
                return f"Metric '{metric}' not found in dataset."
            
            # Perform aggregation
            grouped = self.df.groupby(group_by)[metric].sum().sort_values(ascending=False)
            
            # Format results
            result = f"Results grouped by {group_by}:\n"
            for i, (key, value) in enumerate(grouped.head(10).items(), 1):
                if metric in ['sales', 'profit']:
                    result += f"{i}. {key}: ${value:,.2f}\n"
                else:
                    result += f"{i}. {key}: {value:,.0f}\n"
            
            self.logger.info("Group and aggregate completed", 
                           group_by=group_by, 
                           metric=metric, 
                           groups=len(grouped))
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "Group and aggregate", query=query)
            return f"Error in group and aggregate: {str(e)}"
    
    def _statistical_summary(self, query: str) -> str:
        """Generate statistical summary of the data."""
        self.logger.debug("Generating statistical summary", query=query)
        
        try:
            # Determine which columns to summarize
            columns = []
            if 'sales' in query.lower():
                columns.append('sales')
            if 'profit' in query.lower():
                columns.append('profit')
            if 'quantity' in query.lower():
                columns.append('quantity')
            
            if not columns:
                columns = ['sales', 'profit']  # Default columns
            
            # Filter to existing columns
            columns = [col for col in columns if col in self.df.columns]
            
            if not columns:
                self.logger.warning("No valid columns found for statistical summary")
                return "No valid columns found for statistical summary."
            
            # Generate summary statistics
            summary = self.df[columns].describe()
            
            result = "Statistical Summary:\n"
            for col in columns:
                result += f"\n{col}:\n"
                result += f"  Mean: ${summary.loc['mean', col]:,.2f}\n"
                result += f"  Median: ${summary.loc['50%', col]:,.2f}\n"
                result += f"  Min: ${summary.loc['min', col]:,.2f}\n"
                result += f"  Max: ${summary.loc['max', col]:,.2f}\n"
                result += f"  Std Dev: ${summary.loc['std', col]:,.2f}\n"
            
            self.logger.info("Statistical summary generated", columns=columns)
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "Statistical summary", query=query)
            return f"Error in statistical summary: {str(e)}"
    
    def _time_series_analysis(self, query: str) -> str:
        """Analyze time-based trends."""
        self.logger.debug("Performing time series analysis", query=query)
        
        try:
            if 'order_date' not in self.df.columns:
                self.logger.warning("Date information not available for time series analysis")
                return "Date information not available for time series analysis."
            
            # Determine time grouping
            if 'month' in query.lower():
                time_group = 'month_name'
                time_col = 'month'
            elif 'quarter' in query.lower():
                time_group = 'quarter'
                time_col = 'quarter'
            elif 'year' in query.lower():
                time_group = 'year'
                time_col = 'year'
            else:
                time_group = 'month_name'
                time_col = 'month'
            
            # Determine metric
            metric = 'sales'
            if 'profit' in query.lower():
                metric = 'profit'
            
            if metric not in self.df.columns:
                self.logger.warning("Metric not found in dataset", metric=metric)
                return f"Metric '{metric}' not found in dataset."
            
            # Group by time and calculate trends
            time_data = self.df.groupby([time_group, time_col])[metric].sum().reset_index()
            time_data = time_data.sort_values(time_col)
            
            # Calculate growth rates
            time_data['growth_rate'] = time_data[metric].pct_change() * 100
            
            # Format results
            result = f"Time Series Analysis ({metric} by {time_group}):\n"
            for _, row in time_data.tail(6).iterrows():
                growth = row['growth_rate']
                growth_str = f" ({growth:+.1f}%)" if not pd.isna(growth) else ""
                result += f"{row[time_group]}: ${row[metric]:,.2f}{growth_str}\n"
            
            self.logger.info("Time series analysis completed", 
                           time_group=time_group, 
                           metric=metric, 
                           time_periods=len(time_data))
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "Time series analysis", query=query)
            return f"Error in time series analysis: {str(e)}"
    
    def _correlation_analysis(self, query: str) -> str:
        """Analyze correlations between variables."""
        self.logger.debug("Performing correlation analysis", query=query)
        
        try:
            # Select numeric columns for correlation
            numeric_cols = ['sales', 'profit', 'quantity', 'discount']
            numeric_cols = [col for col in numeric_cols if col in self.df.columns]
            
            if len(numeric_cols) < 2:
                self.logger.warning("Insufficient numeric columns for correlation analysis")
                return "Insufficient numeric columns for correlation analysis."
            
            # Calculate correlation matrix
            corr_matrix = self.df[numeric_cols].corr()
            
            result = "Correlation Analysis:\n"
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicate pairs
                        corr_value = corr_matrix.loc[col1, col2]
                        result += f"{col1} vs {col2}: {corr_value:.3f}\n"
            
            self.logger.info("Correlation analysis completed", variables=numeric_cols)
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "Correlation analysis", query=query)
            return f"Error in correlation analysis: {str(e)}"
    
    def _ranking_analysis(self, query: str) -> str:
        """Perform ranking analysis (top/bottom N)."""
        self.logger.debug("Performing ranking analysis", query=query)
        
        try:
            # Extract number from query
            number_match = re.search(r'(\d+)', query)
            n = int(number_match.group(1)) if number_match else 5
            
            # Determine ranking column
            rank_by = 'sales'
            if 'profit' in query.lower():
                rank_by = 'profit'
            elif 'quantity' in query.lower():
                rank_by = 'quantity'
            
            if rank_by not in self.df.columns:
                self.logger.warning("Ranking column not found", column=rank_by)
                return f"Column '{rank_by}' not found in dataset."
            
            # Determine grouping column
            group_by = 'region'
            if 'product' in query.lower() or 'item' in query.lower():
                group_by = 'product_name' if 'product_name' in self.df.columns else 'category'
            elif 'category' in query.lower():
                group_by = 'category'
            
            if group_by not in self.df.columns:
                self.logger.warning("Grouping column not found", column=group_by)
                return f"Grouping column '{group_by}' not found in dataset."
            
            # Perform ranking
            if 'bottom' in query.lower():
                ranked = self.df.groupby(group_by)[rank_by].sum().nsmallest(n)
                rank_type = "Bottom"
            else:
                ranked = self.df.groupby(group_by)[rank_by].sum().nlargest(n)
                rank_type = "Top"
            
            # Format results
            result = f"{rank_type} {n} {group_by}s by {rank_by}:\n"
            for i, (key, value) in enumerate(ranked.items(), 1):
                if rank_by in ['sales', 'profit']:
                    result += f"{i}. {key}: ${value:,.2f}\n"
                else:
                    result += f"{i}. {key}: {value:,.0f}\n"
            
            self.logger.info("Ranking analysis completed", 
                           rank_type=rank_type, 
                           n=n, 
                           group_by=group_by, 
                           rank_by=rank_by)
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "Ranking analysis", query=query)
            return f"Error in ranking analysis: {str(e)}"
    
    def _profit_margin_analysis(self, query: str) -> str:
        """Perform profit margin analysis."""
        self.logger.debug("Performing profit margin analysis", query=query)
        
        try:
            # Calculate profit margin
            self.df['profit_margin'] = (self.df['profit'] / self.df['sales']) * 100
            self.df['profit_margin'] = self.df['profit_margin'].fillna(0)
            
            # Return profit margin analysis
            result = "Profit Margin Analysis:\n"
            result += f"Mean Profit Margin: {self.df['profit_margin'].mean():.2f}%\n"
            result += f"Median Profit Margin: {self.df['profit_margin'].median():.2f}%\n"
            result += f"Minimum Profit Margin: {self.df['profit_margin'].min():.2f}%\n"
            result += f"Maximum Profit Margin: {self.df['profit_margin'].max():.2f}%\n"
            
            self.logger.info("Profit margin analysis completed")
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "Profit margin analysis", query=query)
            return f"Error in profit margin analysis: {str(e)}"
    
    def _general_analysis(self, query: str) -> str:
        """Perform general analysis when specific type cannot be determined."""
        self.logger.debug("Performing general analysis", query=query)
        
        try:
            # Basic dataset overview
            total_records = len(self.df)
            total_sales = self.df['sales'].sum() if 'sales' in self.df.columns else 0
            total_profit = self.df['profit'].sum() if 'profit' in self.df.columns else 0
            
            result = f"Dataset Overview:\n"
            result += f"Total Records: {total_records:,}\n"
            result += f"Total Sales: ${total_sales:,.2f}\n"
            result += f"Total Profit: ${total_profit:,.2f}\n"
            
            if 'region' in self.df.columns and 'sales' in self.df.columns:
                top_region = self.df.groupby('region')['sales'].sum().idxmax()
                result += f"Top Region by Sales: {top_region}\n"
            
            if 'category' in self.df.columns and 'sales' in self.df.columns:
                top_category = self.df.groupby('category')['sales'].sum().idxmax()
                result += f"Top Category by Sales: {top_category}\n"
            
            self.logger.info("General analysis completed", 
                           total_records=total_records, 
                           total_sales=total_sales, 
                           total_profit=total_profit)
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "General analysis", query=query)
            return f"Error in general analysis: {str(e)}" 