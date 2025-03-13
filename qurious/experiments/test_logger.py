from loguru import logger

logger.info("1. This is a test log message.")
logger.warning("2. This is a test warning message.")

logger.add("test.log", format="{time} {level} {message}", level="INFO")
logger.add("test-w.log", format="{time} {level} {message}", level="INFO", mode="w")
logger.info("3. This message will be logged to test.log.")
logger.remove()

logger.info("4. This message will be logged to test.log.")
