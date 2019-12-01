from .WIKISequence import WIKISequence


class IMDBSequence(WIKISequence):
    def __init__(self, data, target='age', batch_size=16, base_path='../data/raw/imdb_crop/'):
        super().__init__(data, target, batch_size, base_path)

