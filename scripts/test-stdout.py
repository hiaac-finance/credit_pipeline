import logging
import time

# Configure logging to file
logging.basicConfig(filename='output1.txt', 
                    filemode='w',  # Overwrite the file each time the application runs
                    level=logging.DEBUG,  # Capture all levels of logging
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format for the timestamp

logging.debug("hello world"*10)

try:
    for i in range(30):
        if 0/0 == 1:
            pass

except Exception as e:
    k = 4
    logging.exception("An error occurred.")

logging.debug(k)
