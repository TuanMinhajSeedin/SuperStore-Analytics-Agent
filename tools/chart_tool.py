import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
import re
import time
from utils.logger import get_logger

class ChartTool:
    """
    Tool for creating various types of charts and visualizations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the chart tool with a DataFrame.
        
        Args:
            df: Pandas DataFrame containing the data to visualize
        """
        self.df = df.copy()
        self.logger = get_logger("chart_tool")
        self.logger.info("Initializing ChartTool", data_shape=self.df.shape)
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the data for visualization."""
        start_time = time.time()
        self.logger.info("Starting chart data preprocessing")
        
        try:
            # Convert date columns with proper format
            if 'order_date' in self.df.columns:
                # Handle the M/D/YYYY format
                self.df['order_date'] = pd.to_datetime(self.df['order_date'], format='%m/%d/%Y', errors='coerce')
                self.df['year'] = self.df['order_date'].dt.year
                self.df['month'] = self.df['order_date'].dt.month
                self.df['quarter'] = self.df['order_date'].dt.quarter
                self.df['month_name'] = self.df['order_date'].dt.strftime('%B')
                self.logger.info("Date columns processed for visualization")
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['sales', 'profit', 'quantity', 'discount']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    self.logger.debug(f"Converted column to numeric for visualization", column=col)
            
            # Calculate additional metrics
            if 'sales' in self.df.columns and 'profit' in self.df.columns:
                self.df['profit_margin'] = (self.df['profit'] / self.df['sales']) * 100
                self.df['profit_margin'] = self.df['profit_margin'].fillna(0)
                self.logger.info("Profit margin calculated for visualization")
            
            duration = time.time() - start_time
            self.logger.log_performance("Chart data preprocessing", duration, data_shape=self.df.shape)
            
        except Exception as e:
            self.logger.log_error(e, "Chart data preprocessing")
            raise
    
    def create_chart(self, query: str) -> Union[str, go.Figure]:
        """
        Main method to create charts based on the query.
        
        Args:
            query: Chart creation query or instruction
            
        Returns:
            Plotly figure object or error message string
        """
        start_time = time.time()
        self.logger.info("Starting chart creation", query=query)
        
        try:
            query_lower = query.lower()
            
            # Route to appropriate chart creation method
            if any(word in query_lower for word in ['bar', 'column']):
                result = self._create_bar_chart(query)
            elif any(word in query_lower for word in ['line', 'trend', 'time']):
                result = self._create_line_chart(query)
            elif any(word in query_lower for word in ['pie', 'donut']):
                result = self._create_pie_chart(query)
            elif any(word in query_lower for word in ['scatter', 'correlation']):
                result = self._create_scatter_plot(query)
            elif any(word in query_lower for word in ['histogram', 'distribution']):
                result = self._create_histogram(query)
            elif any(word in query_lower for word in ['heatmap']):
                result = self._create_heatmap(query)
            else:
                result = self._create_default_chart(query)
            
            duration = time.time() - start_time
            self.logger.log_performance("Chart creation", duration, query=query, result_type=type(result).__name__)
            
            return result
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, "Chart creation", query=query, duration=duration)
            return f"Error creating chart: {str(e)}"
    
    def _create_bar_chart(self, query: str) -> go.Figure:
        """Create a bar chart based on the query."""
        self.logger.debug("Creating bar chart", query=query)
        
        try:
            # Determine x-axis (categorical variable)
            x_col = None
            query_lower = query.lower()
            if 'product_name' in query_lower and 'product_name' in self.df.columns:
                x_col = 'product_name'
            elif 'product' in query_lower or 'item' in query_lower:
                x_col = 'product_name' if 'product_name' in self.df.columns else 'category'
            elif 'region' in query_lower:
                x_col = 'region'
            elif 'category' in query_lower:
                x_col = 'category'
            elif 'month' in query_lower:
                x_col = 'month_name'
            elif 'quarter' in query_lower:
                x_col = 'quarter'
            elif 'year' in query_lower:
                x_col = 'year'
            
            if not x_col or x_col not in self.df.columns:
                self.logger.warning("Unable to determine x-axis column for bar chart", query=query)
                return px.bar(x=[], y=[], title="Unable to determine x-axis column for bar chart.")
            
            # Determine y-axis (numeric variable)
            y_col = 'sales'
            if 'profit' in query_lower:
                y_col = 'profit'
            elif 'quantity' in query_lower:
                y_col = 'quantity'
            
            if y_col not in self.df.columns:
                self.logger.warning("Y-axis column not found", column=y_col)
                return px.bar(x=[], y=[], title=f"Y-axis column '{y_col}' not found in dataset.")
            
            # Aggregate data
            chart_data = self.df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
            
            # Create bar chart
            fig = px.bar(
                x=chart_data.index,
                y=chart_data.values,
                title=f"{y_col.title()} by {x_col.title()}",
                labels={'x': x_col.title(), 'y': y_col.title()},
                color=chart_data.values,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            
            self.logger.info("Bar chart created successfully", 
                           x_axis=x_col, 
                           y_axis=y_col, 
                           data_points=len(chart_data))
            
            return fig
            
        except Exception as e:
            self.logger.log_error(e, "Bar chart creation", query=query)
            return px.bar(x=[], y=[], title=f"Error creating bar chart: {str(e)}")
    
    def _create_line_chart(self, query: str) -> go.Figure:
        """Create a line chart for time series data."""
        self.logger.debug("Creating line chart", query=query)
        
        try:
            if 'order_date' not in self.df.columns:
                self.logger.warning("Date information not available for line chart")
                return px.line(x=[], y=[], title="Date information not available for line chart.")
            
            # Determine time grouping
            time_col = 'month_name'
            if 'quarter' in query.lower():
                time_col = 'quarter'
            elif 'year' in query.lower():
                time_col = 'year'
            
            # Determine metric
            metric = 'sales'
            if 'profit' in query.lower():
                metric = 'profit'
            elif 'quantity' in query.lower():
                metric = 'quantity'
            
            if metric not in self.df.columns:
                self.logger.warning("Metric not found in dataset", metric=metric)
                return px.line(x=[], y=[], title=f"Metric '{metric}' not found in dataset.")
            
            # Aggregate data by time
            time_data = self.df.groupby(time_col)[metric].sum().reset_index()
            
            # Sort by time if possible
            if time_col == 'month_name':
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                time_data['month_order'] = time_data[time_col].map(lambda x: month_order.index(x) if x in month_order else 999)
                time_data = time_data.sort_values('month_order')
            
            # Create line chart
            fig = px.line(
                x=time_data[time_col],
                y=time_data[metric],
                title=f"{metric.title()} Trend Over Time",
                labels={'x': 'Time Period', 'y': metric.title()},
                markers=True
            )
            
            fig.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            
            self.logger.info("Line chart created successfully", 
                           time_column=time_col, 
                           metric=metric, 
                           time_periods=len(time_data))
            
            return fig
            
        except Exception as e:
            self.logger.log_error(e, "Line chart creation", query=query)
            return px.line(x=[], y=[], title=f"Error creating line chart: {str(e)}")
    
    def _create_pie_chart(self, query: str) -> go.Figure:
        """Create a pie chart for categorical data."""
        self.logger.debug("Creating pie chart", query=query)
        
        try:
            # Determine grouping column
            group_col = 'region'
            if 'category' in query.lower():
                group_col = 'category'
            elif 'product' in query.lower():
                group_col = 'product_name' if 'product_name' in self.df.columns else 'category'
            
            if group_col not in self.df.columns:
                self.logger.warning("Grouping column not found", column=group_col)
                return px.pie(values=[], names=[], title="Grouping column not found in dataset.")
            
            # Determine metric
            metric = 'sales'
            if 'profit' in query.lower():
                metric = 'profit'
            elif 'quantity' in query.lower():
                metric = 'quantity'
            
            if metric not in self.df.columns:
                self.logger.warning("Metric not found in dataset", metric=metric)
                return px.pie(values=[], names=[], title=f"Metric '{metric}' not found in dataset.")
            
            # Aggregate data
            pie_data = self.df.groupby(group_col)[metric].sum().sort_values(ascending=False)
            
            # Create pie chart
            fig = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                title=f"{metric.title()} Distribution by {group_col.title()}",
                hole=0.3  # Make it a donut chart
            )
            
            fig.update_layout(height=500)
            
            self.logger.info("Pie chart created successfully", 
                           group_column=group_col, 
                           metric=metric, 
                           categories=len(pie_data))
            
            return fig
            
        except Exception as e:
            self.logger.log_error(e, "Pie chart creation", query=query)
            return px.pie(values=[], names=[], title=f"Error creating pie chart: {str(e)}")
    
    def _create_scatter_plot(self, query: str) -> go.Figure:
        """Create a scatter plot for correlation analysis."""
        self.logger.debug("Creating scatter plot", query=query)
        
        try:
            # Determine x and y variables
            x_col = 'sales'
            y_col = 'profit'
            
            if 'quantity' in query.lower():
                x_col = 'quantity'
            elif 'discount' in query.lower():
                x_col = 'discount'
            
            if x_col not in self.df.columns or y_col not in self.df.columns:
                self.logger.warning("Required columns not found", x_col=x_col, y_col=y_col)
                return px.scatter(x=[], y=[], title="Required columns not found in dataset.")
            
            # Create scatter plot
            fig = px.scatter(
                x=self.df[x_col],
                y=self.df[y_col],
                title=f"{y_col.title()} vs {x_col.title()}",
                labels={'x': x_col.title(), 'y': y_col.title()},
                color=self.df['profit_margin'] if 'profit_margin' in self.df.columns else None,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=500)
            
            # Calculate correlation
            correlation = self.df[x_col].corr(self.df[y_col])
            
            self.logger.info("Scatter plot created successfully", 
                           x_axis=x_col, 
                           y_axis=y_col, 
                           correlation=correlation,
                           data_points=len(self.df))
            
            return fig
            
        except Exception as e:
            self.logger.log_error(e, "Scatter plot creation", query=query)
            return px.scatter(x=[], y=[], title="Error creating scatter plot.")
    
    def _create_histogram(self, query: str) -> go.Figure:
        """Create a histogram for distribution analysis."""
        self.logger.debug("Creating histogram", query=query)
        
        try:
            # Determine variable to plot
            col = 'sales'
            if 'profit' in query.lower():
                col = 'profit'
            elif 'quantity' in query.lower():
                col = 'quantity'
            elif 'discount' in query.lower():
                col = 'discount'
            
            if col not in self.df.columns:
                self.logger.warning("Column not found for histogram", column=col)
                return px.histogram(x=[], title="Column not found for histogram.")
            
            # Create histogram
            fig = px.histogram(
                x=self.df[col],
                title=f"Distribution of {col.title()}",
                labels={'x': col.title(), 'y': 'Count'},
                nbins=30
            )
            
            fig.update_layout(height=500)
            
            # Calculate statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            
            self.logger.info("Histogram created successfully", 
                           column=col, 
                           mean=mean_val, 
                           median=median_val,
                           data_points=len(self.df))
            
            return fig
            
        except Exception as e:
            self.logger.log_error(e, "Histogram creation", query=query)
            return px.histogram(x=[], title="Error creating histogram.")
    
    def _create_heatmap(self, query: str) -> go.Figure:
        """Create a heatmap for correlation matrix."""
        self.logger.debug("Creating heatmap", query=query)
        
        try:
            # Select numeric columns
            numeric_cols = ['sales', 'profit', 'quantity', 'discount']
            numeric_cols = [col for col in numeric_cols if col in self.df.columns]
            
            if len(numeric_cols) < 2:
                self.logger.warning("Insufficient numeric columns for heatmap")
                return px.imshow(z=[[]], title="Insufficient numeric columns for heatmap.")
            
            # Calculate correlation matrix
            corr_matrix = self.df[numeric_cols].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            fig.update_layout(height=500)
            
            self.logger.info("Heatmap created successfully", 
                           variables=numeric_cols, 
                           matrix_size=f"{len(numeric_cols)}x{len(numeric_cols)}")
            
            return fig
            
        except Exception as e:
            self.logger.log_error(e, "Heatmap creation", query=query)
            return px.imshow(z=[[]], title="Error creating heatmap.")
    
    def _create_default_chart(self, query: str) -> go.Figure:
        """Create a default chart when specific type cannot be determined."""
        self.logger.debug("Creating default chart", query=query)
        
        try:
            # Create a simple bar chart of sales by region
            if 'region' in self.df.columns and 'sales' in self.df.columns:
                region_sales = self.df.groupby('region')['sales'].sum().sort_values(ascending=False)
                
                fig = px.bar(
                    x=region_sales.index,
                    y=region_sales.values,
                    title="Sales by Region",
                    labels={'x': 'Region', 'y': 'Sales ($)'}
                )
                
                fig.update_layout(height=500)
                
                self.logger.info("Default chart created successfully", 
                               chart_type="bar", 
                               x_axis="Region", 
                               y_axis="Sales",
                               regions=len(region_sales))
                
                return fig
            else:
                self.logger.warning("Required columns not available for default chart")
                return px.bar(x=[], y=[], title="Unable to create default chart due to missing required columns.")
                
        except Exception as e:
            self.logger.log_error(e, "Default chart creation", query=query)
            return px.bar(x=[], y=[], title=f"Error creating default chart: {str(e)}")
    
    def create_dashboard(self, metrics: List[str] = None) -> str:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            metrics: List of metrics to include in dashboard
            
        Returns:
            Dashboard description
        """
        start_time = time.time()
        self.logger.info("Creating comprehensive dashboard", metrics=metrics)
        
        try:
            if metrics is None:
                metrics = ['sales', 'profit', 'region', 'category']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sales by Region', 'Profit by Category', 'Sales Trend', 'Profit Distribution'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "histogram"}]]
            )
            
            # Chart 1: Sales by Region
            if 'region' in self.df.columns and 'sales' in self.df.columns:
                region_sales = self.df.groupby('region')['sales'].sum()
                fig.add_trace(
                    go.Bar(x=region_sales.index, y=region_sales.values, name="Sales by Region"),
                    row=1, col=1
                )
                self.logger.debug("Added sales by region chart to dashboard")
            
            # Chart 2: Profit by Category
            if 'category' in self.df.columns and 'profit' in self.df.columns:
                category_profit = self.df.groupby('category')['profit'].sum()
                fig.add_trace(
                    go.Bar(x=category_profit.index, y=category_profit.values, name="Profit by Category"),
                    row=1, col=2
                )
                self.logger.debug("Added profit by category chart to dashboard")
            
            # Chart 3: Sales Trend
            if 'order_date' in self.df.columns and 'sales' in self.df.columns:
                monthly_sales = self.df.groupby(self.df['order_date'].dt.to_period('M'))['sales'].sum()
                fig.add_trace(
                    go.Scatter(x=monthly_sales.index.astype(str), y=monthly_sales.values, name="Sales Trend"),
                    row=2, col=1
                )
                self.logger.debug("Added sales trend chart to dashboard")
            
            # Chart 4: Profit Distribution
            if 'profit' in self.df.columns:
                fig.add_trace(
                    go.Histogram(x=self.df['profit'], name="Profit Distribution"),
                    row=2, col=2
                )
                self.logger.debug("Added profit distribution chart to dashboard")
            
            fig.update_layout(height=800, title_text="SuperStore Analytics Dashboard")
            
            duration = time.time() - start_time
            self.logger.log_performance("Dashboard creation", duration, charts=4)
            
            dashboard_description = "Created comprehensive dashboard with 4 charts: Sales by Region, Profit by Category, Sales Trend over time, and Profit Distribution histogram."
            
            return dashboard_description
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, "Dashboard creation", duration=duration)
            return f"Error creating dashboard: {str(e)}" 