from matplotlib import pyplot as plot

class Individual:
    def __init__(self, im_matrix_array):
        self._im_matrix_array = im_matrix_array

    def show(self):
        fig = plot.figure(figsize=(5, 5))
        columns = 5
        rows = 2
        for i in range(1, columns*rows +1):
            fig.add_subplot(rows, columns, i)
            plot.imshow(self._im_matrix_array[i-1], cmap="Greys_r")
        plot.show()