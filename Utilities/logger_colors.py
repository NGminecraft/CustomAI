import logging

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: '\033[32m', # Green
        logging.INFO: '\033[34m', # Blue
        logging.WARNING: '\033[93m', # Yellow
        logging.ERROR: '\033[31m', # Red
        logging.CRITICAL: '\033[31m\033[47m', # Red on White
    }
    
    RESET = '\033[0m' # Reset color

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"