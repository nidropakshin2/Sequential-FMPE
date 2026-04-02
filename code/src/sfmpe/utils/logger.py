import logging
import sys
from typing import Optional, Union
from datetime import datetime
import os


class Logger:
    """
    Custom logger for SFMPE project with file and console output support.
    """
    
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    def __init__(
        self,
        name: str = "sfmpe",
        log_level: int = INFO,
        log_to_file: bool = False,
        log_file_path: Optional[str] = None,
        log_to_console: bool = True,
        format_string: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to a file
            log_file_path: Path to log file (if None, generates default path)
            log_to_console: Whether to log to console
            format_string: Custom format string for log messages
        """
        self.name = name
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Remove existing handlers
        
        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            if log_file_path is None:
                # Create logs directory if it doesn't exist
                logs_dir = "logs"
                os.makedirs(logs_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = os.path.join(logs_dir, f"{name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file_path = log_file_path
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def log(self, level: int, message: str, *args, **kwargs):
        """Log message at specified level."""
        self.logger.log(level, message, *args, **kwargs)
    
    def set_level(self, level: int):
        """Set logging level."""
        self.log_level = level
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def get_log_file_path(self) -> Optional[str]:
        """Get path to log file if file logging is enabled."""
        return getattr(self, 'log_file_path', None)
    
    def progress(
        self,
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        decimals: int = 1,
        length: int = 50,
        fill: str = "█"
    ):
        """
        Print a progress bar.
        
        Args:
            current: Current progress
            total: Total steps
            prefix: Prefix string
            suffix: Suffix string
            decimals: Number of decimals in percent
            length: Character length of bar
            fill: Bar fill character
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        
        message = f'\r{prefix} |{bar}| {percent}% {suffix}'
        
        # Use info level for progress bar
        if current == total:
            self.info(message.strip())
        else:
            # For incomplete progress, we need to handle carriage return specially
            # We'll use a simpler approach for logging
            self.debug(f"{prefix} Progress: {percent}% {suffix}")


def get_default_logger() -> Logger:
    """
    Get a default logger instance.
    
    Returns:
        Logger: Default logger instance
    """
    return Logger(
        name="sfmpe",
        log_level=Logger.INFO,
        log_to_file=False,
        log_to_console=True
    )


# Convenience functions for quick logging
def setup_logging(
    name: str = "sfmpe",
    level: int = Logger.INFO,
    log_to_file: bool = False,
    log_file_path: Optional[str] = None
) -> Logger:
    """
    Quick setup for logging.
    
    Returns:
        Logger: Configured logger instance
    """
    return Logger(
        name=name,
        log_level=level,
        log_to_file=log_to_file,
        log_file_path=log_file_path,
        log_to_console=True
    )