import json


class PipeData:
    def __init__(self, data_sequence, batch_size=1, shuffle=False) -> None:
        self.data_sequence = data_sequence
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_item(self):
        pass

    def iter(self):
        pass 
