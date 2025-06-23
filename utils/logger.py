import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional

class AppLogger:
    """
    Application logger with file and console output, rotation, and structured formatting.
    """
    
    def __init__(self, name: str = "analytical_agent", log_level: str = "INFO"):
        """
        Initialize the application logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with file and console handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # File handler with rotation
        log_file = self.logs_dir / "application.logs"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with additional context."""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {context}"
        return message
    
    def log_query(self, query: str, user_id: Optional[str] = None):
        """Log user query with context."""
        self.info("User query received", query=query, user_id=user_id)
    
    def log_analysis_start(self, query: str, analysis_type: str):
        """Log start of analysis."""
        self.info("Analysis started", query=query, analysis_type=analysis_type)
    
    def log_analysis_complete(self, query: str, duration: float, success: bool):
        """Log completion of analysis."""
        self.info("Analysis completed", 
                 query=query, 
                 duration=f"{duration:.2f}s", 
                 success=success)
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log error with context."""
        self.error(f"Error in {context}: {str(error)}", 
                  error_type=type(error).__name__, 
                  **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.info(f"Performance: {operation}", duration=f"{duration:.3f}s", **kwargs)
    
    def log_data_operation(self, operation: str, data_shape: tuple, **kwargs):
        """Log data operations."""
        self.info(f"Data operation: {operation}", 
                 rows=data_shape[0], 
                 columns=data_shape[1], 
                 **kwargs)

# Global logger instance
logger = AppLogger()

def get_logger(name: str = None) -> AppLogger:
    """
    Get a logger instance.
    
    Args:
        name: Optional logger name
        
    Returns:
        AppLogger instance
    """
    if name:
        return AppLogger(name)
    return logger 