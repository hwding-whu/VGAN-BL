from src import logger


if __name__ == '__main__':
    test_logger = logger.Logger('test')
    test_logger.debug('This is DEBUG test.')
    test_logger.info('This is INFO test.')
    print(test_logger.level)
    src.config.config.level = 'INFO'
    test_logger = logger.Logger('test')
    print(test_logger.level)
    test_logger.debug('You shouldn\'t see me.')
