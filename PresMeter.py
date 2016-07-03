import numpy as np
import GridOut as go
import FlowCalc as fc
import CrowdArt as ca
import CrowdCount as cc
import Functions as fn
import TrainSVM as ts
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

#DIR_PATH = '/home/pol/PFM_Data/Frames/angels.frames.ini/angels.frames'
DIR_PATH = '/home/pol/PFM_Data/Frames/Kamera05_1620_1640.mp4.frames'
# MATRIX_PATH = '/home/pol/PFM_Data/Matrix/matrix.2010.data'
MATRIX_PATH = '/home/pol/PFM_Data/Matrix/matrix.data'
#IMG_NAME = '/angels.frame'
IMG_NAME = '/Kamera05_1620_1640.mp4.frame'
# IMG_NAME = '/test_'
# TEST_DIR = '/home/pol/PFM_Data/tests'
IMG_FORMAT = '.png'
IMG_DIR = '/home/pol/PFM_Data/Images'
# TEST_DIR = '/home/pol/PFM_Data/tests'

#TRAINING_DB = '/home/pol/Dropbox/Master Modelling/PFM/Python Code/CrowdCount/TRAINING_DB/training_db_head.txt'
TRAINING_DB = '/home/pol/Dropbox/Master Modelling/PFM/Python Code/CrowdCount/TRAINING_DB/training_specific_alem.txt'

omega = 0.8
alpha_value = 0.001

# number of subcell for the computation an graphic representation
x_square = 10
y_square = 10

number_of_trans = 10#80

frame_step = 6

number_initial_frame = 1

kernel_x = np.array([[-1.0, 1.0],
                     [-1.0, 1.0]])

kernel_y = np.array([[-1.0, -1.0],
                     [1.0, 1.0]])

range_step = 5

if __name__ == '__main__':

    '''
    fc.calculate_flow(MATRIX_PATH, number_initial_frame, number_of_trans, frame_step, omega, kernel_x, kernel_y,
                      alpha_value)
    '''

    frames = np.arange(number_initial_frame, number_initial_frame + (number_of_trans * frame_step), frame_step)
    for i, frame in enumerate(frames):
        cCount = cc.CrowdCount(frame)
        FS_values = cCount.calculate_npeople()
        #FS_values = np.delete(FS_values, 2, 2)

        tSVM = ts.TrainSVM(TRAINING_DB, range_step)
        #tSVM.plot_database()

        SVM_values = tSVM.predict_values(FS_values)

        print '\rIteration density: ' + str(i + 1) + '/' + str(number_of_trans) + ' done',

        cCount.save_data(MATRIX_PATH, frame, SVM_values)  # cCount.convert_to_size(SVM_values))

        # cCount.save_data(MATRIX_PATH, frame, cCount.convert_to_size(FS_values))
        # cArt = ca.CrowdArt(cCount.get_original_image(), SVM_values, range_step)

    print

    '''
    fc.generate_pressure(MATRIX_PATH, number_initial_frame, number_of_trans, frame_step, omega)
    '''
    grid_out = go.GridOut()
    grid_out.print_output()

