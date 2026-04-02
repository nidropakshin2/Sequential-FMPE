import sys
import os

from sfmpe.utils.logger import Logger, get_default_logger, setup_logging


def test_demonstrate_logger():
    """Demonstrate the Logger class functionality."""
    
    print("=== Demonstrating Logger Class ===\n")
    
    # 1. Create a default logger
    print("1. Default logger:")
    default_logger = get_default_logger()
    default_logger.info("This is an info message from default logger")
    default_logger.debug("This is a debug message (might not appear due to level)")
    default_logger.warning("This is a warning message")
    
    # 2. Create a custom logger
    print("\n2. Custom logger with different settings:")
    custom_logger = Logger(
        name="test_logger",
        log_level=Logger.DEBUG,
        log_to_file=False,
        log_to_console=True
    )
    custom_logger.debug("Debug message - now visible!")
    custom_logger.info("Info message")
    custom_logger.error("Error message")
    
    # 3. Demonstrate progress bar
    print("\n3. Progress bar demonstration:")
    custom_logger.info("Starting simulation...")
    total_steps = 10
    for i in range(total_steps):
        custom_logger.progress(
            current=i + 1,
            total=total_steps,
            prefix="Simulation",
            suffix=f"Step {i+1}/{total_steps}"
        )
    
    # 4. File logging example
    print("\n4. File logging (check logs/ directory):")
    file_logger = setup_logging(
        name="file_logger",
        level=Logger.INFO,
        log_to_file=True,
        log_file_path=f"{os.path.dirname(__file__)}/logs/test_example.log"
    )
    file_logger.info("This message goes to both console and file")
    file_logger.info(f"Log file: {file_logger.get_log_file_path()}")
    
    print("\n=== Logger demonstration complete ===")

    assert os.path.exists(file_logger.get_log_file_path()), "Log file was not created as expected"

# if __name__ == "__main__":
#     demonstrate_logger()