import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # LangChain Configuration
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "")
    
    # Application Configuration
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Data Configuration
    DATA_PATH = os.getenv("DATA_PATH", "artifacts/SuperStoreOrders.csv")
    
    # Chart Configuration
    CHART_HEIGHT = int(os.getenv("CHART_HEIGHT", "500"))
    CHART_WIDTH = int(os.getenv("CHART_WIDTH", "800"))
    
    # Logging Configuration
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    LOG_FILE = os.getenv("LOG_FILE", "application.logs")
    LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    LOG_FORMAT = os.getenv("LOG_FORMAT", "detailed")  # "detailed" or "simple"
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in your environment "
                "variables or create a .env file with OPENAI_API_KEY=your_key_here"
            )
        
        if not os.path.exists(cls.DATA_PATH):
            raise FileNotFoundError(
                f"Data file not found at {cls.DATA_PATH}. "
                "Please ensure the SuperStoreOrders.csv file exists in the artifacts folder."
            )
    
    @classmethod
    def get_openai_config(cls):
        """Get OpenAI configuration dictionary."""
        return {
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE,
            "openai_api_key": cls.OPENAI_API_KEY
        }
    
    @classmethod
    def get_logging_config(cls):
        """Get logging configuration dictionary."""
        return {
            "log_level": cls.LOG_LEVEL,
            "log_dir": cls.LOG_DIR,
            "log_file": cls.LOG_FILE,
            "log_max_size": cls.LOG_MAX_SIZE,
            "log_backup_count": cls.LOG_BACKUP_COUNT,
            "log_format": cls.LOG_FORMAT
        } 