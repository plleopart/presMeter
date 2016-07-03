import ctypes
import cv2
from scipy.ndimage.interpolation import zoom
import numpy as np
import PresMeter as mw
import matplotlib.pyplot as plt

ones_matrix = np.array([[1.0, 1.0],
                        [1.0, 1.0]])
minus_ones_matrix = np.array([[-1.0, -1.0],
                              [-1.0, -1.0]])


def get_frame(i):
    # this function returns the "i" image from given path
    # print "Reading frame",
    # print i
    cap = cv2.imread(mw.DIR_PATH + mw.IMG_NAME + str(i) + mw.IMG_FORMAT)
    img = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    # return a matrix as numpy.array

    if img.shape[1] > 1000 or img.shape[0] > 1000:
        img = zoom(img, 0.4)

    return img[0:(int((img.shape[0] / mw.x_square)) * mw.x_square),
           0:(int((img.shape[1] / mw.y_square)) * mw.y_square)].astype(float)


def reduce_matrix(input_matrix):
    output_matrix = np.zeros((mw.x_square, mw.y_square), dtype=np.double)
    fl = ctypes.cdll.LoadLibrary('./_flowc.so')
    r_matrix = fl.reduce_matrix
    r_matrix(ctypes.c_void_p(input_matrix.ctypes.data), ctypes.c_int(input_matrix.shape[0]),
             ctypes.c_int(input_matrix.shape[1]), ctypes.c_int(mw.x_square), ctypes.c_int(mw.y_square),
             ctypes.c_void_p(output_matrix.ctypes.data))
    return output_matrix


def check_nan_val(matrix, id):
    for val in matrix:
        for v in val:
            if np.isnan(v):
                print str(id) + ": exit"
                exit()


def draw_frames():
    for frame in range(1000):
        i = frame
        # frame_step = 6
        # omega = 1.43
        try:

            press = np.load(mw.MATRIX_PATH + '/pressure_matrix' + str(frame) + '.npy')

            plt.imshow(press)
            # plt.gray()
            plt.savefig(mw.IMG_DIR + '/pressure' + str(frame) + '.png')
            plt.clf()
            # <--------------------------------------------------------------------------------------------------------->

            # exit()

            print '\rDone ' + str(frame),
        except IOError:
            pass  # print "File " + str(frame) + " does not exist"
    print


def LK_art(frame_actual, frame_next, velocity_u, velocity_v, win_size, kernel_x, kernel_y, alpha_value, id):
    # np.seterr(divide='ignore', invalid='ignore')
    # start = timeit.default_timer()

    frame_actual = frame_actual / 255.0
    frame_next = frame_next / 255.0


    '''
    if frame_actual.shape[1] > 1000:
        frame_actual = zoom(frame_actual, 0.4)
        plt.gray()
        plt.imsave('/home/pol/fo/Kamera05_1620_1640.mp4.frame' + str(id) + '.png', frame_actual * 255.0)
        frame_next = zoom(frame_next, 0.4)
    '''
    '''
    if frame_actual.shape[1] > 1000:
        frame_actual = zoom(frame_actual, 0.4)
    '''

    h_value = 1.0

    # print "Initializing matrix"
    # initialize partials
    partial_Ix = np.zeros((frame_actual.shape[0], frame_actual.shape[1]), dtype=np.double)
    partial_Iy = np.zeros((frame_actual.shape[0], frame_actual.shape[1]), dtype=np.double)
    partial_It_I = np.zeros((frame_actual.shape[0], frame_actual.shape[1]), dtype=np.double)
    partial_It_II = np.zeros((frame_actual.shape[0], frame_actual.shape[1]), dtype=np.double)

    # initialize output velocities
    # velocity_u = np.zeros((frame_actual.shape[0], frame_actual.shape[1]), dtype=np.double)
    # velocity_v = np.zeros((frame_actual.shape[0], frame_actual.shape[1]), dtype=np.double)

    fl = ctypes.cdll.LoadLibrary('./_flowc.so')

    # here we compute the convolutions
    # 1: input_matrix, 2: input_kernel, 3: number of rows, 4: number of cols, 5: number of kernel rows, 6: number of
    # kernel cols, 7: output_matrix
    convol2d = fl.convol_2d
    convol2d(ctypes.c_void_p(frame_actual.ctypes.data), ctypes.c_void_p(kernel_x.ctypes.data),
             ctypes.c_int(frame_actual.shape[0]),
             ctypes.c_int(frame_actual.shape[1]), ctypes.c_int(kernel_x.shape[0]), ctypes.c_int(kernel_x.shape[1]),
             ctypes.c_void_p(partial_Ix.ctypes.data))
    convol2d(ctypes.c_void_p(frame_actual.ctypes.data), ctypes.c_void_p(kernel_y.ctypes.data),
             ctypes.c_int(frame_actual.shape[0]),
             ctypes.c_int(frame_actual.shape[1]), ctypes.c_int(kernel_y.shape[0]), ctypes.c_int(kernel_y.shape[1]),
             ctypes.c_void_p(partial_Iy.ctypes.data))
    convol2d(ctypes.c_void_p(frame_actual.ctypes.data), ctypes.c_void_p(ones_matrix.ctypes.data),
             ctypes.c_int(frame_actual.shape[0]),
             ctypes.c_int(frame_actual.shape[1]), ctypes.c_int(ones_matrix.shape[0]),
             ctypes.c_int(ones_matrix.shape[1]),
             ctypes.c_void_p(partial_It_I.ctypes.data))
    convol2d(ctypes.c_void_p(frame_next.ctypes.data), ctypes.c_void_p(minus_ones_matrix.ctypes.data),
             ctypes.c_int(frame_next.shape[0]),
             ctypes.c_int(frame_next.shape[1]), ctypes.c_int(minus_ones_matrix.shape[0]),
             ctypes.c_int(minus_ones_matrix.shape[1]),
             ctypes.c_void_p(partial_It_II.ctypes.data))

    # sum partials t I and II
    partial_It = partial_It_I + partial_It_II

    # print "Starting flow algorithm"
    # here we iterate over the velocities
    # 1: partial_x, 2: partial_y, 3: partial_t, 4: velocity_u, 5: velocity_v, 6: x counter,
    # 7: y counter, 8: h value, 9: alpha value, 10: omega, 11: output_u, 12: output_v
    velocity_iterator = fl.velocity_iteration

    av_x = []
    av_y = []
    x_range = []

    for i in range(150):
        # print "\rIteration: " + str(i + 1) + " done",
        velocity_iterator(ctypes.c_void_p(partial_Ix.ctypes.data), ctypes.c_void_p(partial_Iy.ctypes.data),
                          ctypes.c_void_p(partial_It.ctypes.data), ctypes.c_void_p(velocity_u.ctypes.data),
                          ctypes.c_void_p(velocity_v.ctypes.data),
                          ctypes.c_int(frame_actual.shape[0]), ctypes.c_int(frame_actual.shape[1]),
                          ctypes.c_double(h_value),
                          ctypes.c_double(alpha_value), ctypes.c_double(win_size),
                          ctypes.c_void_p(velocity_u.ctypes.data),
                          ctypes.c_void_p(velocity_v.ctypes.data))

        av_x.append(np.average(np.abs(velocity_u)))
        av_y.append(np.average(np.abs(velocity_v)))
        x_range.append(i)
    # print

    # stop = timeit.default_timer()

    # print "Time inside",
    # print stop - start

    '''
    # <-------------------------------------->
    # with this we plot the velocity
    plt.plot(x_range, av_x, color="blue")
    plt.plot(x_range, av_y, color="red")
    # plt.ylim([0.0, 0.050])
    # plt.xlim([0, 200])
    # plt.ylim([-0.0065, 0.0065])
    plt.grid()
    #plt.savefig("Outputs/plt-" + str(win_size) + "_" + str(
    #        id) + '-' + str(alpha_value) + ".png")
    plt.show()
    '''
    plt.clf()
    # <-------------------------------------->

    velocity = []
    '''
    print "max"
    print velocity_u.max()
    print velocity_v.max()
    print 'average'
    print np.average(np.abs(velocity_u))
    '''
    velocity.append(velocity_u)
    velocity.append(velocity_v)


    return velocity


def Smooth_gauss(frame_actual):
    '''
    # R value
    R = 1.0

    # frame_actual = np.asarray(frame_actual)

    # frame_actual = np.random.rand(100, 200)

    vector_smoothed_field = np.zeros((frame_actual.shape[0], frame_actual.shape[1]), dtype=np.double)

    # smooth the input with shape:
    #       .
    #    .  .  .
    # .  .  0  .  .
    #    .  .  .
    #       .
    # the following is a 5x5 matrix with values according to,
    # 1/pi*R^2*exp(-abs(r(t)-r)^2/R)

    # the matrix is just the part exp(-abs(r(t)-r)),
    # after we will calculate the entire matrix

    distance_matrix = np.array([[0.0, 0.0, 0.14, 0.0, 0.0],
                                [0.0, 0.25, 0.36, 0.25, 0.0],
                                [0.14, 0.36, 1.0, 0.36, 0.14],
                                [0.0, 0.25, 0.36, 0.25, 0.0],
                                [0.0, 0.0, 0.14, 0.0, 0.0]])

    f_func = (1 / (np.pi * R ** 2)) * distance_matrix ** (1 / R ** 2)

    # this value is the sum of all elements of the distance_matrix, we use this in order to calculate the average
    sum_f_func = np.sum(f_func)
    '''
    fl = ctypes.cdll.LoadLibrary('./_flowc.so')
    convol2d = fl.convol_2d
    '''
    frame_actual = frame_actual / 1.0

    convol2d(ctypes.c_void_p(frame_actual.ctypes.data), ctypes.c_void_p(distance_matrix.ctypes.data),
             ctypes.c_int(frame_actual.shape[0]),
             ctypes.c_int(frame_actual.shape[1]), ctypes.c_int(distance_matrix.shape[0]),
             ctypes.c_int(distance_matrix.shape[1]),
             ctypes.c_void_p(vector_smoothed_field.ctypes.data))

    '''
    input_matrix = np.array([[4.0, 1.0, -7.0],
                            [-2.0, 3.0, 9.0],
                            [9.0, -5.0, 5.0]])

    kernel_matrix = np.array([[2.0, 1.0, -3.0],
                            [9.0, 1.0, 4.0,],
                            [-6.0, 7.0, -3.0]])
    output_matrix = np.zeros((3, 3))

    convol2d(ctypes.c_void_p(input_matrix.ctypes.data), ctypes.c_void_p(kernel_matrix.ctypes.data),
             ctypes.c_int(input_matrix.shape[0]),
             ctypes.c_int(input_matrix.shape[1]), ctypes.c_int(kernel_matrix.shape[0]),
             ctypes.c_int(kernel_matrix.shape[1]),
             ctypes.c_void_p(output_matrix.ctypes.data))
    print output_matrix

    #return vector_smoothed_field / sum_f_func
