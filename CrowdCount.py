import numpy as np
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.fftpack import fft2, ifft2
import cv2
import Functions as fn
import PresMeter as mw
import matplotlib.pyplot as plt

# DIR_PATH = '/home/pol/PFM_Data/Frames/angels.frames'
# IMG = '/angels.frame1'
# IMG_FORMAT = '.png'

# CALIBRATE_PATH = '/home/pol/Dropbox/Master Modelling/PFM/Python Code/CrowdCount/CALIBRATE_FIG'

# head_size = 8.0
head_err = 0.52


# sig = (0.27 * head_size) + 0.71


class CrowdCount:
    def __init__(self, img):

        self.crowded_image = fn.get_frame(img)

        self.delta_x = int(self.crowded_image.shape[1] / mw.x_square)
        self.delta_y = int(self.crowded_image.shape[0] / mw.y_square)

        self.x_range = np.arange(self.delta_x, self.crowded_image.shape[1] + self.delta_x, self.delta_x)
        self.y_range = np.arange(self.delta_y, self.crowded_image.shape[0] + self.delta_y, self.delta_y)

        self.crowded_image = self.crowded_image[0: self.delta_y * mw.y_square, 0:self.delta_x * mw.x_square]

        self.altitude = len(self.y_range)

        self.count_matrix = np.zeros((len(self.x_range), len(self.y_range), 3), dtype=int)

        self.total_SIFT = 0
        self.total_FOUR = 0

        self.xi = 0
        self.yi = 0

    def __use_head_size(self):
        '''
        head_array = []
        for alt in range(self.altitude):
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            plt.gray()
            plt.imshow(self.crowded_image[alt * self.delta_y:(alt * self.delta_y) + self.delta_y,
                       0:self.crowded_image.shape[1]])
            plt.show()

            head_array.append(
                    float(raw_input('Number of pixels per head ' + str(alt + 1) + '/' + str(self.altitude) + ':\n')))

            plt.clf()

        print 'Head array:',
        print head_array
        '''
        #head_array = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # angels
        head_array = [1, 2, 2, 3, 3, 4, 4, 6, 6, 8]  # alem

        return head_array

    def calculate_npeople(self):
        heads = self.__use_head_size()
        for i, xr in enumerate(self.x_range):
            self.yi = 0
            for j, yc in enumerate(self.y_range):
                sub_crowded_image = self.crowded_image[self.yi:yc, self.xi:xr]
                head_size = heads[i]
                sig = (0.27 * head_size) + 0.71
                '''
                if i == 2:
                    if j == 5:

                        sift = cv2.SIFT()
                        kp = sift.detect(sub_crowded_image.astype(np.uint8), None)

                        img=cv2.drawKeypoints(sub_crowded_image.astype(np.uint8),kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        #cv2.imshow('sift_keypoints.jpg',img)
                        plt.imshow(img)
                        plt.show()

                        reduced_kp = []
                        for keypoint in kp:
                            if head_size - (head_size * head_err) < keypoint.size < head_size + head_err:
                                reduced_kp.append(keypoint)

                        img=cv2.drawKeypoints(sub_crowded_image.astype(np.uint8),reduced_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        plt.imshow(img)
                        plt.show()

                        exit()

                        reduced_kp = []
                        for keypoint in kp:
                            if head_size - (head_size * head_err) < keypoint.size < head_size + head_err:
                                reduced_kp.append(keypoint)

                        sift_local_points = len(reduced_kp)
                        self.total_SIFT += sift_local_points






                        plt.gray()
                        plt.imshow(sub_crowded_image)
                        plt.show()

                        sub_crowded_image = fft2(sub_crowded_image)

                        sub_crowded_image = self.__del_row(4, sub_crowded_image)
                        sub_crowded_image = self.__del_col(4, sub_crowded_image)

                        sub_crowded_image = ifft2(sub_crowded_image)
                        sub_crowded_image = np.abs(sub_crowded_image)

                        plt.imshow(sub_crowded_image)
                        plt.show()

                        sub_crowded_image = gaussian_filter(sub_crowded_image, sig, mode='constant')

                        plt.imshow(sub_crowded_image)
                        plt.show()

                        detected_peaks = self.__detect_peaks(sub_crowded_image)

                        four_local_points = np.sum(detected_peaks)

                        plt.imshow(detected_peaks)
                        plt.show()

                        exit()

                '''
                # <------------------------------------------------------------------------------------------------>
                # this part count the number of point through using SIFT analysis
                sift = cv2.SIFT()
                kp = sift.detect(sub_crowded_image.astype(np.uint8), None)

                reduced_kp = []
                for keypoint in kp:
                    if head_size - (head_size * head_err) < keypoint.size < head_size + head_err:
                        reduced_kp.append(keypoint)

                sift_local_points = len(reduced_kp)
                self.total_SIFT += sift_local_points


                # <------------------------------------------------------------------------------------------------>
                # This part count the number of points by using Fourier analysis

                sub_crowded_image = fft2(sub_crowded_image)

                sub_crowded_image = self.__del_row(4, sub_crowded_image)
                sub_crowded_image = self.__del_col(4, sub_crowded_image)

                sub_crowded_image = ifft2(sub_crowded_image)
                sub_crowded_image = np.abs(sub_crowded_image)

                sub_crowded_image = gaussian_filter(sub_crowded_image, sig, mode='constant')

                detected_peaks = self.__detect_peaks(sub_crowded_image)

                four_local_points = np.sum(detected_peaks)

                self.total_FOUR += four_local_points

                self.count_matrix[i][j][0] = four_local_points
                self.count_matrix[i][j][1] = sift_local_points
                self.count_matrix[i][j][2] = head_size


                self.yi = yc
            self.xi = xr
        return self.count_matrix

    def get_original_image(self):
        return self.crowded_image

    def save_data(self, path, frame, data):
        '''
        for y in range(data.shape[1]):
            for x in range(data.shape[0]):
                if data[x][y] == 0:
                    data[x][y] = 1.0
        '''
        np.save(path + '/matrix_density_' + str(frame) + '.npy', data)

    def convert_to_size(self, reduced_matrix):
        big_matrix = np.zeros(self.crowded_image.shape)
        for x in range(self.crowded_image.shape[1]):
            for y in range(self.crowded_image.shape[0]):
                big_matrix[y][x] = reduced_matrix[(y / self.delta_y)][(x / self.delta_x)]  # [1]  # 0:fourier, 1:sift
        return big_matrix

    def __del_row(self, number, matrix):
        if number < 0:
            number = np.abs(number)
            for row in range(number):
                matrix[-row - 1, :] = 0.0

        else:
            for row in range(number):
                matrix[row, :] = 0.0

        return matrix

    def __del_col(self, number, matrix):
        if number < 0:
            number = np.abs(number)
            for col in range(number):
                matrix[:, -col - 1] = 0.0

        else:
            for col in range(number):
                matrix[:, col] = 0.0

        return matrix

    def __detect_peaks(self, image):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max = maximum_filter(image, (4, 4)) == image

        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (image == 0)

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask
        detected_peaks = local_max - eroded_background

        return detected_peaks
