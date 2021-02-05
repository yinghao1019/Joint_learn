import logging
logging.basicConfig(filename='test.log',
                    format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%d/%m/%Y', level=logging.INFO)
logging.info('hello world')
