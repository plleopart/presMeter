import PresMeter as mw
import numpy as np
import Functions as fn
import CrowdArt as ca


class GridOut:
    def __init__(self):
        counter = 0
        first = True

        self.frames_p = []
        self.frames_v = []
        self.frames_d = []

        self.frames_number = []

        ma_p = []
        mi_p = []

        ma_v = []
        mi_v = []

        ma_d = []
        mi_d = []

        frame = 0

        while counter < (mw.frame_step * 2) + 1:
            frame += 1

            try:
                if first:
                    p = np.load(mw.MATRIX_PATH + '/pressure_matrix' + str(frame) + '.npy')
                    '''
                    v = np.load(
                            mw.MATRIX_PATH + '/matrix_x_vel_' + str(frame) + '-' + str(
                                    frame + mw.frame_step) + '_' + str(
                                    mw.omega) + '.npy')

                    d = np.load(mw.MATRIX_PATH + '/matrix_density_' + str(frame) + '.npy')
                    '''
                    self.delta_x = int(p.shape[1] / mw.x_square)
                    self.delta_y = int(p.shape[0] / mw.y_square)

                    self.x_range = np.arange(self.delta_x, p.shape[1], self.delta_x)
                    self.y_range = np.arange(self.delta_y, p.shape[0], self.delta_y)
                    first = False

                press = (np.load(mw.MATRIX_PATH + '/pressure_matrix' + str(frame) + '.npy'))

                v_x = np.load(
                        mw.MATRIX_PATH + '/matrix_x_vel_' + str(frame) + '-' + str(
                                frame + mw.frame_step) + '_' + str(
                                mw.omega) + '.npy')
                v_y = np.load(
                        mw.MATRIX_PATH + '/matrix_y_vel_' + str(frame) + '-' + str(
                                frame + mw.frame_step) + '_' + str(
                                mw.omega) + '.npy')

                vel = np.sqrt(v_x ** 2 + v_y ** 2)

                '''
                for y in range(v_m.shape[0]):
                    for x in range(v_m.shape[1]):
                        if v_m[y][x] < 0.3:
                            v_m[y][x] = 0.0
                '''

                den = (np.load(mw.MATRIX_PATH + '/matrix_density_' + str(frame) + '.npy'))

                ma_p.append(press.max())
                mi_p.append(press.min())

                ma_d.append(den.max())
                mi_d.append(den.min())

                ma_v.append(vel.max())
                mi_v.append(vel.min())

                self.frames_p.append(press)
                self.frames_v.append(vel)
                self.frames_d.append(den)

                self.frames_number.append(frame)
                counter = 0

            except IOError:
                counter += 1

        self.p_min = min(mi_p)
        self.p_max = max(ma_p)

        self.d_min = min(mi_d)
        self.d_max = max(ma_d)

        self.v_min = min(mi_v)
        self.v_max = max(ma_v)

    def print_output(self):
        for i, frame_name in enumerate(self.frames_number):
            img = fn.get_frame(frame_name)
            ca.CrowdArt(img, self.frames_p[i], 0, self.p_min, self.p_max, frame_name, 'Alem_P/pressure')
            ca.CrowdArt(img, self.frames_v[i], 0, self.v_min, self.v_max, frame_name, 'Alem_V/velocity')
            ca.CrowdArt(img, self.frames_d[i], 0, self.d_min, self.d_max, frame_name, 'Alem_D/density')

            print '\r Frame ' + str(i + 1) + ' of ' + str(len(self.frames_number)) + ' saved',

    '''
    def __reformat_matrix(self, frame):
        self.reform_frames = np.zeros((len(self.x_range), len(self.y_range)))
        xi = 0
        for i, xr in enumerate(self.x_range):
            yi = 0
            for j, yc in enumerate(self.y_range):
                self.reform_frames[i][j] = np.average(frame[yi:yc, xi:xr])

                yi = yc
            xi = xr

        # self.frames[f] = self.reform_frames
        return self.reform_frames
    '''
