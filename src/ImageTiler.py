import numpy as np

class ImageTiler:
    def __init__(self, image, tile_width, tile_height):
        self.image = image
        self.__n_columns = int(image.shape[1] / tile_width)
        self.__n_rows = int(image.shape[0] / tile_height)

    def getTiles(self):
        width = int(self.image.shape[0] / self.__n_rows)
        height = int(self.image.shape[1] / self.__n_columns)
        tiles = []
        for i in range(self.__n_rows):
            for j in range(self.__n_columns):
                tiles.append(self.image[i * width:(i + 1) * width, j * height:(j + 1) * height])
        return tiles

    def reassembleImage(self, tiles):
        width = int(self.image.shape[0] / self.__n_rows)
        height = int(self.image.shape[1] / self.__n_columns)
        res_image = np.zeros(self.image.shape)
        for i in range(self.__n_rows):
            for j in range(self.__n_columns):
                res_image[i * width:(i + 1) * width, j * height:(j + 1) * height] = tiles[i * self.__n_columns + j]
        return res_image

