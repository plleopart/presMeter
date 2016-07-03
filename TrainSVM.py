import numpy as np
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

'''
HEAD_ON = True
F = 37
S = 34
REAL = 27
H = 7
'''


class TrainSVM:
    def __init__(self, database, range_step):
        # here we use database to train our algorithm
        f = open(database, 'r')
        self.range_step = range_step
        self.mTraining_db = []

        for n_line, line in enumerate(f.readlines()):
            if n_line != 0:
                i = 0
                f = 0
                sample = []

                while f != len(line) - 1:
                    while not ';' in line[f]:
                        f += 1
                    sample.append(int(line[i:f]))
                    f += 1
                    i = f
                self.mTraining_db.append(sample)
            else:
                pass

        self.mTraining_db = np.asarray(self.mTraining_db)
        self.mTraining_db = self.mTraining_db.T
        self.labels_db = self.mTraining_db[2]
        self.mTraining_db = np.delete(self.mTraining_db, 2, 0)

        self.mTraining_db = self.mTraining_db.T

        self.labels_db_range = []
        for label in self.labels_db:
            i = 0
            f = range_step
            while not i <= label < f:
                i += range_step
                f += range_step
                if f > self.labels_db.max() + range_step:
                    raise ValueError('Label out of range...')

            self.labels_db_range.append(i)

        if len(self.labels_db_range) != self.mTraining_db.shape[0]:
            raise ValueError('Values do not match...')

        self.clf = svm.SVC(kernel='linear', C=1.0)
        self.clf.fit(self.mTraining_db, self.labels_db_range)

    def plot_database(self):
        # if we want to plot data in a 3D-plot
        fig = pylab.figure()
        ax = Axes3D(fig)

        self.mTraining_db = self.mTraining_db.T
        ax.scatter(self.mTraining_db[0], self.mTraining_db[1], self.labels_db)

        ax.set_xlabel('Fourier Parameter')
        ax.set_ylabel('SIFT Parameter')
        ax.set_zlabel('Real Count')

        ax.set_xlim(0, self.mTraining_db[0].max())
        ax.set_ylim(0, self.mTraining_db[1].max())
        ax.set_zlim(0, self.labels_db.max())

        pyplot.show()

    def predict_value(self, F, S, H):
        # taking values from (F)ourier and (S)ift as input we return the prediction as string
        return self.clf.predict([[F, S, H]])

    def predict_values(self, values):
        prediction_matrix = np.zeros((values.shape[0], values.shape[1]), dtype=int)
        for x in range(values.shape[0]):
            for y in range(values.shape[1]):
                prediction_matrix[x][y] = self.clf.predict([values[x][y]])
        return prediction_matrix
