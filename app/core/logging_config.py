import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from typing import Optional

# Enable ANSI colors on Windows terminals
if sys.platform == "win32":
    os.system("")

# Root log directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ANALYSES_LOGS_DIR = os.path.join(LOGS_DIR, "analyses")

# Ensure log directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ANALYSES_LOGS_DIR, exist_ok=True)

# Context variables to track active User and Analysis across async calls
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="N/A")
current_analysis_id: ContextVar[str] = ContextVar("current_analysis_id", default="N/A")


def set_log_context(user_id: Optional[str] = None, analysis_id: Optional[str] = None):
    """
    Set active user and analysis context variables for log enrichment.
    """
    if user_id is not None:
        current_user_id.set(str(user_id))
    if analysis_id is not None:
        current_analysis_id.set(str(analysis_id))


def clear_log_context():
    """
    Reset active user and analysis context variables to default 'N/A'.
    """
    current_user_id.set("N/A")
    current_analysis_id.set("N/A")


class ContextFilter(logging.Filter):
    """
    Filter to automatically inject user_id and analysis_id into every log record.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = current_user_id.get()
        record.analysis_id = current_analysis_id.get()
        return True


class LevelFilter(logging.Filter):
    """
    Filter to include only log records at or above min_level.
    """
    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


class AgentFilter(logging.Filter):
    """
    Filter to capture agent-related executions, inputs/outputs, tool calls, costs, claims, and context building.
    """
    AGENT_PREFIXES = (
        "app.agents",
        "app.tracing",
        "app.indexing",
        "app.api.claims",
        "app.api.context",
        "agent",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name.lower()
        msg = record.getMessage().lower()
        if any(name.startswith(prefix.lower()) for prefix in self.AGENT_PREFIXES):
            return True
        keywords = ("agent", "tool", "claim", "span", "token", "cost", "qdrant", "enrichment", "extraction")
        if any(kw in name or kw in msg for kw in keywords):
            return True
        return False


class ColoredConsoleFormatter(logging.Formatter):
    """
    ANSI Color Formatter for Console Output:
    - INFO: GREEN (\033[92m)
    - WARNING: YELLOW (\033[93m)
    - ERROR / CRITICAL: RED (\033[91m)
    - DEBUG: CYAN (\033[96m)
    """
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[96m",            # Cyan
        logging.INFO: "\033[92m",             # Green
        logging.WARNING: "\033[93m",          # Yellow
        logging.ERROR: "\033[91m",            # Red
        logging.CRITICAL: "\033[91m\033[1m", # Bold Red
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        user = getattr(record, "user_id", "N/A")
        analysis = getattr(record, "analysis_id", "N/A")

        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_str = f"{color}{record.levelname:<7}{self.RESET}"
        user_str = f"\033[35m[User: {user}]\033[0m"
        analysis_str = f"\033[34m[Analysis: {analysis}]\033[0m"

        return f"[{timestamp}] {level_str} [{record.name}] {user_str} {analysis_str} - {record.getMessage()}"


class DynamicAnalysisHandler(logging.Handler):
    """
    Dynamic file handler that segregates logs into dedicated files per user & analysis:
    logs/analyses/user_{user_id}_analysis_{analysis_id}.log
    """
    def emit(self, record: logging.LogRecord):
        user_id = getattr(record, "user_id", "N/A")
        analysis_id = getattr(record, "analysis_id", "N/A")

        if analysis_id == "N/A" and user_id == "N/A":
            return

        safe_user = str(user_id).replace("@", "_at_").replace("/", "_").replace("\\", "_")
        safe_analysis = str(analysis_id).replace("/", "_").replace("\\", "_")
        filename = f"user_{safe_user}_analysis_{safe_analysis}.log"
        filepath = os.path.join(ANALYSES_LOGS_DIR, filename)

        try:
            if self.formatter:
                log_line = self.format(record) + "\n"
            else:
                from datetime import datetime
                timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"[{timestamp}] [{record.levelname}] [{record.name}] [User: {user_id}] [Analysis: {analysis_id}] - {record.getMessage()}\n"

            with open(filepath, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO):
    """
    Initialize central Evidentia logging configuration with colored console output,
    app.log, error.log, agents.log, user_analysis.log, and per-analysis log files.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear pre-existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    context_filter = ContextFilter()

    # 1. Colored Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredConsoleFormatter())
    console_handler.addFilter(context_filter)
    root_logger.addHandler(console_handler)

    # Standard File Formatter
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [User: %(user_id)s] [Analysis: %(analysis_id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 2. General app.log Handler (ALL logs)
    app_log_path = os.path.join(LOGS_DIR, "app.log")
    app_handler = RotatingFileHandler(app_log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(file_formatter)
    app_handler.addFilter(context_filter)
    root_logger.addHandler(app_handler)

    # 3. error.log Handler (WARNING and ERROR logs ONLY)
    error_log_path = os.path.join(LOGS_DIR, "error.log")
    error_handler = RotatingFileHandler(error_log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(file_formatter)
    error_handler.addFilter(context_filter)
    error_handler.addFilter(LevelFilter(logging.WARNING))
    root_logger.addHandler(error_handler)

    # 4. agents.log Handler (Agent inputs, outputs, tool calls, costs, claims, context)
    agents_log_path = os.path.join(LOGS_DIR, "agents.log")
    agents_handler = RotatingFileHandler(agents_log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    agents_handler.setLevel(level)
    agents_handler.setFormatter(file_formatter)
    agents_handler.addFilter(context_filter)
    agents_handler.addFilter(AgentFilter())
    root_logger.addHandler(agents_handler)

    # 5. user_analysis.log Handler (Logs sorted/demarcated by user & analysis)
    user_analysis_path = os.path.join(LOGS_DIR, "user_analysis.log")
    user_analysis_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [USER: %(user_id)s] [ANALYSIS: %(analysis_id)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    user_analysis_handler = RotatingFileHandler(user_analysis_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    user_analysis_handler.setLevel(level)
    user_analysis_handler.setFormatter(user_analysis_formatter)
    user_analysis_handler.addFilter(context_filter)
    root_logger.addHandler(user_analysis_handler)

    # 6. Dynamic Per-Analysis Log File Handler
    dynamic_handler = DynamicAnalysisHandler()
    dynamic_handler.setLevel(level)
    dynamic_handler.setFormatter(file_formatter)
    dynamic_handler.addFilter(context_filter)
    root_logger.addHandler(dynamic_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)

    root_logger.info("Evidentia Centralized Logging System Initialized.")
