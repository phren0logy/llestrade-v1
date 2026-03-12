#!/usr/bin/env python3
"""
Test script for verifying the logging and exception handling implementation.
Tests basic logging functionality, exception capture, and crash reporting.
"""

import sys
import logging
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLabel

from src.config.logging_config import ApplicationLogger
from src.config.paths import app_logs_dir, app_crashes_dir
from src.core.exception_handler import GlobalExceptionHandler


class LoggingTestWindow(QWidget):
    """Test window for exception handling demonstration."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the test UI."""
        self.setWindowTitle("Logging & Exception Test")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("Click buttons to test logging and exception handling:")
        layout.addWidget(info_label)
        
        # Test buttons
        test_log_btn = QPushButton("Test Logging Levels")
        test_log_btn.clicked.connect(self.test_logging_levels)
        layout.addWidget(test_log_btn)
        
        test_warning_btn = QPushButton("Generate Warning")
        test_warning_btn.clicked.connect(self.test_warning)
        layout.addWidget(test_warning_btn)
        
        test_error_btn = QPushButton("Generate Handled Error")
        test_error_btn.clicked.connect(self.test_handled_error)
        layout.addWidget(test_error_btn)
        
        test_crash_btn = QPushButton("Generate Unhandled Exception (Crash)")
        test_crash_btn.clicked.connect(self.test_unhandled_exception)
        layout.addWidget(test_crash_btn)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        
        self.setLayout(layout)
        
    def test_logging_levels(self):
        """Test different logging levels."""
        self.logger.debug("This is a DEBUG message - detailed information")
        self.logger.info("This is an INFO message - general information")
        self.logger.warning("This is a WARNING message - something to pay attention to")
        self.logger.error("This is an ERROR message - something went wrong")
        self.logger.critical("This is a CRITICAL message - serious problem")
        
        self.log_display.append("Tested all logging levels - check console and log file")
        
    def test_warning(self):
        """Generate a warning scenario."""
        self.logger.warning("API rate limit approaching (80% used)")
        self.log_display.append("Generated warning - check logs")
        
    def test_handled_error(self):
        """Test handled error with proper logging."""
        try:
            # Simulate an error
            result = 10 / 0
        except ZeroDivisionError as e:
            self.logger.error(
                "Division by zero error occurred",
                exc_info=True,
                extra={
                    'operation': 'test_calculation',
                    'values': {'numerator': 10, 'denominator': 0}
                }
            )
            self.log_display.append("Handled error logged - check logs for full traceback")
            
    def test_unhandled_exception(self):
        """Generate an unhandled exception to test crash handling."""
        self.log_display.append("Generating unhandled exception in 2 seconds...")
        # Give user time to see the message
        QApplication.processEvents()
        time.sleep(2)
        
        # This will cause an unhandled exception
        raise RuntimeError("Test crash - this is an intentional unhandled exception!")


def test_logging_without_gui():
    """Test logging functionality without GUI."""
    logger = logging.getLogger("test_module")
    
    print("\n=== Testing logging without GUI ===")
    logger.info("Starting non-GUI tests")
    
    # Test module-specific logging
    llm_logger = logging.getLogger("src.common.llm.providers")
    llm_logger.debug("Testing LLM provider debug logging")
    llm_logger.info("Testing LLM provider info logging")
    
    # Test log file location
    log_dir = app_logs_dir()
    print(f"\nLog files location: {log_dir}")
    
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log*"))
        print(f"Found {len(log_files)} log file(s):")
        for log_file in log_files:
            print(f"  - {log_file.name} ({log_file.stat().st_size} bytes)")
    
    logger.info("Non-GUI tests completed")


def main():
    """Main test function."""
    # Parse arguments
    debug_mode = "--debug" in sys.argv
    no_gui = "--no-gui" in sys.argv
    
    # Setup logging
    logger_config = ApplicationLogger()
    logger_config.setup(debug=debug_mode)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting logging and exception handling tests")
    
    if no_gui:
        # Run non-GUI tests only
        test_logging_without_gui()
    else:
        # Run GUI tests
        app = QApplication(sys.argv)
        app.setApplicationName("Logging Test App")
        
        # Install exception handler
        crash_dir = app_crashes_dir()
        exception_handler = GlobalExceptionHandler(crash_dir)
        exception_handler.install()
        
        # Show crash directory info
        print(f"\nCrash reports location: {crash_dir}")
        
        # Create and show test window
        window = LoggingTestWindow()
        window.show()
        
        # Run app
        exit_code = app.exec()
        
        # Cleanup
        exception_handler.uninstall()
        logger.info("Test application exiting")
        
        sys.exit(exit_code)


if __name__ == "__main__":
    print("Logging and Exception Handling Test Script")
    print("Usage:")
    print("  python test_logging_and_exceptions.py           # Run with GUI")
    print("  python test_logging_and_exceptions.py --no-gui  # Run without GUI")
    print("  python test_logging_and_exceptions.py --debug   # Enable debug logging")
    print()
    
    main()
