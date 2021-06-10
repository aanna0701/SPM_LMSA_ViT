from collections import OrderedDict

class Logger_dict():
    def __init__(self):
        self.dict = OrderedDict()
    
    def append(self, key, value):
        self.dict[key] = value
        
    def print(self, logger):
        logger.debug(self.dict)