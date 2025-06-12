import os
import glob
import logging

class Logger:
    """
    用于输出训练日志并保存日志文件
    """

    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'train_log.txt')
        logging.basicConfig(
            filename=log_path,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.console_handler)

    def info(self, msg):
        self.logger.info(msg)