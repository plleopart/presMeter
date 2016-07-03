import numpy as np
import Functions as fn
import PresMeter as mw
import ctypes
import warnings


def calculate_flow(MATRIX_PATH, number_initial_frame, number_of_trans, frame_step, omega, kernel_x, kernel_y,
                   alpha_value):

    print "Generating vector matrix..."

    frames = np.arange(number_initial_frame, number_initial_frame + (number_of_trans * frame_step), frame_step)

    for i, frame in enumerate(frames):
        frame_actual = fn.get_frame(frame)
        frame_next = fn.get_frame(frame + frame_step)

        if i == 0:
            velocity_u = np.zeros(frame_actual.shape, dtype=np.double)
            velocity_v = np.zeros(frame_actual.shape, dtype=np.double)
        else:
            velocity_u = LK[0]
            velocity_v = LK[1]

        LK = fn.LK_art(frame_actual, frame_next, velocity_u, velocity_v, omega, kernel_x, kernel_y, alpha_value,
                       frame)

        print "\rMatrix " + str(frame) + "-" + str(frame + frame_step) + " calculated. " + str(i + 1) + "/" + str(
                len(frames)),

        np.save(
                MATRIX_PATH + '/matrix_x_vel_' + str(frame) + '-' + str(frame + frame_step) + '_' + str(
                        omega) + '.npy',
                fn.reduce_matrix(LK[0]))
        np.save(
                MATRIX_PATH + '/matrix_y_vel_' + str(frame) + '-' + str(frame + frame_step) + '_' + str(
                        omega) + '.npy',
                fn.reduce_matrix(LK[1]))


def generate_pressure(MATRIX_PATH, number_initial_frame, number_of_trans, frame_step, omega):
    print "Generating pressure matrix..."

    frames = np.arange(number_initial_frame, number_initial_frame + (number_of_trans * frame_step), frame_step)
    document_on = True
    for i, frame in enumerate(frames):
        var_distance = 2

        if i >= var_distance:
            if document_on:
                # here we have the average density between two transitions
                density_actual = np.load(MATRIX_PATH + '/matrix_density_' + str(frame) + '.npy')
                density_next = np.load(MATRIX_PATH + '/matrix_density_' + str(frame + frame_step) + '.npy')

                average_density = (density_actual + density_next) / 2

                row = average_density.shape[0]
                col = average_density.shape[1]

                var_velocity_matrix_x = np.zeros([row, col])
                var_velocity_matrix_y = np.zeros([row, col])
                # this is an array with 5 values
                var_frames = np.arange(frame - (var_distance * frame_step),
                                       frame + ((var_distance + 1) * frame_step),
                                       frame_step)

                frames_x = []
                frames_y = []
                position_actual_frame = var_frames[2]

                for i in var_frames:
                    try:

                        x_loc = np.load(MATRIX_PATH + '/matrix_x_vel_' + str(i) + '-' + str(i + frame_step) + '_' + str(
                                omega) + '.npy')

                        y_loc = np.load(MATRIX_PATH + '/matrix_y_vel_' + str(i) + '-' + str(i + frame_step) + '_' + str(
                                omega) + '.npy')


                    except IOError:
                        document_on = False

                    frames_x.append(x_loc)
                    frames_y.append(y_loc)

                fl = ctypes.cdll.LoadLibrary('./_flowc.so')
                calc_var = fl.variance
                calc_var(ctypes.c_void_p(frames_x[0].ctypes.data), ctypes.c_void_p(frames_x[1].ctypes.data),
                         ctypes.c_void_p(frames_x[2].ctypes.data), ctypes.c_void_p(frames_x[3].ctypes.data),
                         ctypes.c_void_p(frames_x[4].ctypes.data), ctypes.c_int(density_actual.shape[0]),
                         ctypes.c_int(density_actual.shape[1]), ctypes.c_void_p(var_velocity_matrix_x.ctypes.data))
                calc_var(ctypes.c_void_p(frames_y[0].ctypes.data), ctypes.c_void_p(frames_y[1].ctypes.data),
                         ctypes.c_void_p(frames_y[2].ctypes.data), ctypes.c_void_p(frames_y[3].ctypes.data),
                         ctypes.c_void_p(frames_y[4].ctypes.data), ctypes.c_int(density_actual.shape[0]),
                         ctypes.c_int(density_actual.shape[1]), ctypes.c_void_p(var_velocity_matrix_y.ctypes.data))

                module_velocity_matrix = (
                    (np.sqrt(((var_velocity_matrix_x) ** 2) + ((var_velocity_matrix_y) ** 2))))
                pressure_matrix = module_velocity_matrix * average_density
                # print 'MAX press',
                # print pressure_matrix.max()

                np.save(MATRIX_PATH + '/pressure_matrix' + str(position_actual_frame) + '.npy', pressure_matrix)
                # np.save(TEST_DIR + '/pressure_matrix_log' + str(position_actual_frame) + '.npy', pressure_matrix)
                print "\rMatrix pressure " + str(frame) + "-" + str(frame + frame_step) + " saved",
            else:
                print
                print 'No such file or directory:',
                print MATRIX_PATH + '/matrix_x_vel_' + str(i) + '-' + str(i + frame_step) + '_' + str(
                        omega) + '.npy'
                print 'No more documents'


def reduce_matrix(big_m):
    # this is used to avoid np.average of zeros
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        delta_x = int(big_m.shape[1] / mw.x_square)
        delta_y = int(big_m.shape[0] / mw.y_square)

        xi = 0
        reduced_m = np.zeros((mw.x_square, mw.y_square))
        for xr in range(big_m.shape[1]):
            yi = 0
            for yc in range(big_m.shape[0]):
                if int(yc / delta_y) < mw.y_square and int(xr / delta_x) < mw.x_square:
                    reduced_m[int(yc / delta_y)][int(xr / delta_x)] = np.average(big_m[yi:yc, xi:xr])

                yi = yc
            xi = xr
        return reduced_m
