from PIL import Image, ImageDraw, ImageFont
import PresMeter as mw
import Functions as fn

alpha_u = 50
alpha_a = 200


class CrowdArt:
    def __init__(self, img, values, range_step, g_min, g_max, id, name):
        self.values = values
        self.range_step = range_step
        self.x_range = img.shape[1]
        self.y_range = img.shape[0]

        self.g_min = g_min
        self.g_max = g_max

        self.crowd_img = Image.fromarray(img)
        self.crowd_img = self.crowd_img.convert('RGB')

        self.first_x = self.x_range / self.values.shape[1]
        self.first_y = self.y_range / self.values.shape[0]

        self.draw = ImageDraw.Draw(self.crowd_img, 'RGBA')

        self.__fill_surface()
        self.__write_count()
        self.__draw_grid()

        # self.crowd_img.show()

        self.crowd_img.crop((0, 0, self.first_x * self.values.shape[1], self.y_range)).save(
                mw.IMG_DIR + '/' + str(name) + str(id) + '.png')

        # self.crowd_img.save(mw.IMG_DIR + '/pressure' + str(id) + '.png')

    def __draw_grid(self):
        for x in range(self.x_range / self.first_x):
            self.draw.line((x * self.first_x, 0, x * self.first_x, self.y_range), width=2, fill='black')
        for y in range(self.y_range / self.first_y):
            self.draw.line((0, y * self.first_y, self.x_range, y * self.first_y), width=2, fill='black')

    def __write_count(self):
        f_type = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-LI.ttf", 12)

        for y in range(self.values.shape[0]):
            for x in range(self.values.shape[1]):
                '''
                self.draw.text(((self.first_x * x) + self.first_x / 6, (self.first_y * y) + self.first_y / 2),
                               str(self.values[y][x]) + '-' + str(self.values[y][x] + self.range_step),
                               font=f_type, fill='blue')
                '''
                self.draw.text(((self.first_x * x) + self.first_x / 6, (self.first_y * y) + self.first_y / 2),
                               '%.2f' % self.values[y][x],
                               font=f_type, fill='blue')

    def __fill_surface(self):
        # minim = self.values.min()
        # maxim = self.values.max()

        minim = self.g_min
        maxim = self.g_max

        for y in range(self.values.shape[0]):
            for x in range(self.values.shape[1]):
                self.draw.rectangle(
                        ((x * self.first_x, y * self.first_y), ((x + 1) * self.first_x, (y + 1) * self.first_y)),
                        fill=(255, 0, 0, self.__map_value(self.values[y][x], minim, maxim, alpha_u, alpha_a)))

    def __map_value(self, val, oldMin, oldMax, newMin, newMax):
        # this function map a value val from old range to new range
        x = (float((val - oldMin)) / float((oldMax - oldMin))) * float((newMax - newMin)) + float(newMin)
        # return mapped value
        return int(x)
