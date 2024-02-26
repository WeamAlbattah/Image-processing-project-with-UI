import math
from email.mime import image
import numpy as np
import PIL
from PIL import Image
from PyQt5 import Qt
from PyQt5.QtGui import QPixmap


#
## Guasiian filter
#
def apply_gaussian_filter(image):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    filter_size = 5
    matrix = [1, 4, 7, 4, 1, 4, 20, 33, 20, 4, 7, 33, 55, 33, 7, 4, 20, 33, 20, 4, 1, 4, 7, 4, 1]
    matrix_sum = sum(matrix)

    for x in range(filter_size // 2, width - filter_size // 2):
        for y in range(filter_size // 2, height - filter_size // 2):
            total_r, total_g, total_b = 0, 0, 0
            index = 0

            for i in range(-filter_size // 2, filter_size // 2 + 1):
                for j in range(-filter_size // 2, filter_size // 2 + 1):
                    pixel = image.getpixel((x + i, y + j))
                    total_r += pixel[0] * matrix[index]
                    total_g += pixel[1] * matrix[index]
                    total_b += pixel[2] * matrix[index]
                    index += 1

            average_r = total_r // matrix_sum
            average_g = total_g // matrix_sum
            average_b = total_b // matrix_sum

            output_image.putpixel((x, y), (average_r, average_g, average_b))

    return output_image

#
## Median filter
#

def apply_median_filter(image, filter_size):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    half_filter_size = (filter_size - 1) // 2

    for x in range(half_filter_size, width - half_filter_size):
        for y in range(half_filter_size, height - half_filter_size):
            r_values, g_values, b_values = [], [], []

            for i in range(-half_filter_size, half_filter_size + 1):
                for j in range(-half_filter_size, half_filter_size + 1):
                    pixel = image.getpixel((x + i, y + j))
                    r_values.append(pixel[0])
                    g_values.append(pixel[1])
                    b_values.append(pixel[2])

            r_values.sort()
            g_values.sort()
            b_values.sort()

            median_r = r_values[len(r_values) // 2]
            median_g = g_values[len(g_values) // 2]
            median_b = b_values[len(b_values) // 2]

            output_image.putpixel((x, y), (median_r, median_g, median_b))

    return output_image
filter_size=5

#
## netlestirme filter
#
def apply_netlestirme_filter(image):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    filter_size = 2
    matrix = [0, -2, 0, -2, 11, -2, 0, -2, 0]
    matrix_sum = sum(matrix)

    for x in range(filter_size // 2, width - filter_size // 2):
        for y in range(filter_size // 2, height - filter_size // 2):
            total_r, total_g, total_b = 0, 0, 0

            for i in range(-filter_size // 2, filter_size // 2 + 1):
                for j in range(-filter_size // 2, filter_size // 2 + 1):
                    new_x = x + i
                    new_y = y + j
                    if 0 <= new_x < width and 0 <= new_y < height:
                        pixel = image.getpixel((new_x, new_y))
                        if isinstance(pixel, (tuple, list)) and len(pixel) >= 3:
                            total_r += pixel[0] * matrix[(i + filter_size // 2) * 3 + (j + filter_size // 2)]
                            total_g += pixel[1] * matrix[(i + filter_size // 2) * 3 + (j + filter_size // 2)]
                            total_b += pixel[2] * matrix[(i + filter_size // 2) * 3 + (j + filter_size // 2)]

            r = total_r // matrix_sum
            g = total_g // matrix_sum
            b = total_b // matrix_sum

            r = max(0, min(r, 255))  # Clamp values between 0 and 255
            g = max(0, min(g, 255))
            b = max(0, min(b, 255))

            output_image.putpixel((x, y), (r, g, b))

    return output_image

#
## prewitt filter
#
def apply_prewitt_filter(image):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    filter_size = 3

    for x in range(filter_size // 2, width - filter_size // 2):
        for y in range(filter_size // 2, height - filter_size // 2):
            p1 = image.getpixel((x - 1, y - 1))[0]
            p2 = image.getpixel((x, y - 1))[0]
            p3 = image.getpixel((x + 1, y - 1))[0]
            p4 = image.getpixel((x - 1, y))[0]
            p5 = image.getpixel((x, y))[0]
            p6 = image.getpixel((x + 1, y))[0]
            p7 = image.getpixel((x - 1, y + 1))[0]
            p8 = image.getpixel((x, y + 1))[0]
            p9 = image.getpixel((x + 1, y + 1))[0]

            gx = abs(-p1 + p3 - p4 + p6 - p7 + p9)
            gy = abs(p1 + p2 + p3 - p7 - p8 - p9)
            prewitt_value = gx + gy

            output_image.putpixel((x, y), (prewitt_value, prewitt_value, prewitt_value))

    return output_image

#
## laplace filter
#
def apply_laplace_filter(image):
    width, height = image.size  # Access image size from the provided image object
    output_image = Image.new("RGB", (width, height))
    bm1 = Image.new("RGB", (width, height))

    for i in range(1, width - 1):
        for j in range(1, height - 1):
            renk2 = image.getpixel((i, j - 1))
            renk4 = image.getpixel((i - 1, j))
            renk5 = image.getpixel((i, j))
            renk6 = image.getpixel((i + 1, j))
            renk8 = image.getpixel((i, j + 1))
            color_red = renk2[0] + renk4[0] + renk5[0] * (-4) + renk6[0] + renk8[0]
            color_green = renk2[1] + renk4[1] + renk5[1] * (-4) + renk6[1] + renk8[1]
            color_blue = renk2[2] + renk4[2] + renk5[2] * (-4) + renk6[2] + renk8[2]
            average = (color_red + color_green + color_blue) // 3
            if average > 255:
                average = 255
            if average < 0:
                average = 0
            bm1.putpixel((i, j), (average, average, average))

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            pixel1 = image.getpixel((x, y))
            pixel2 = bm1.getpixel((x, y))
            r = pixel1[0] + pixel2[0]
            g = pixel1[1] + pixel2[1]
            b = pixel1[2] + pixel2[2]
            if r > 255:
                r = 255
            if g > 255:
                g = 255
            if b > 255:
                b = 255
            if r < 0:
                r = 0
            if g < 0:
                g = 0
            if b < 0:
                b = 0
            output_image.putpixel((x, y), (r, g, b))

    return output_image

#
## negative filter
#
def apply_negative_filter(image):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            r = 255 - pixel[0]
            g = 255 - pixel[1]
            b = 255 - pixel[2]
            output_image.putpixel((x, y), (r, g, b))

    return output_image

#
## Sobel filter
#
def apply_sobel_filter(image):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    filter_size = 3

    for x in range(filter_size // 2, width - filter_size // 2):
        for y in range(filter_size // 2, height - filter_size // 2):
            p1 = image.getpixel((x - 1, y - 1))[0]
            p2 = image.getpixel((x, y - 1))[0]
            p3 = image.getpixel((x + 1, y - 1))[0]
            p4 = image.getpixel((x - 1, y))[0]
            p5 = image.getpixel((x, y))[0]
            p6 = image.getpixel((x + 1, y))[0]
            p7 = image.getpixel((x - 1, y + 1))[0]
            p8 = image.getpixel((x, y + 1))[0]
            p9 = image.getpixel((x + 1, y + 1))[0]

            gx = abs(-p1 + p3 - 2 * p4 + 2 * p6 - p7 + p9)
            gy = abs(p1 + 2 * p2 + p3 - p7 - 2 * p8 - p9)
            gxy = gx + gy

            output_image.putpixel((x, y), (gxy, gxy, gxy))

    return output_image

#
## ters Cevirme
#
def apply_terscevir_filter(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Input image must be a PIL Image object.")

    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    angle = 180  # Degrees for rotation
    radians = angle * math.pi / 180
    center_x = width // 2
    center_y = height // 2

    for x in range(width):
        for y in range(height):
            input_pixel = image.getpixel((x, y))
            output_x = int(x)
            output_y = int(-y + 2 * center_y)

            if 0 <= output_x < width and 0 <= output_y < height:
                output_image.putpixel((output_x, output_y), input_pixel)

    return output_image

#
## uzaklastirma filter
#
def uzaklastir(input_image):
    output_image = Image.new("RGB", input_image.size)
    scaling_factor = 2

    for x1 in range(0, input_image.width, scaling_factor):
        for y1 in range(0, input_image.height, scaling_factor):
            pixel_color = input_image.getpixel((x1, y1))
            output_image.putpixel((x1 // scaling_factor, y1 // scaling_factor), pixel_color)

    return output_image

#
## aynalama
#
def apply_aynalama(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Input image must be a PIL Image object.")

    ResimGenisligi = image.width
    ResimYuksekligi = image.height
    CikisResmi = Image.new("RGB", (ResimGenisligi, ResimYuksekligi))  # Create the output image

    Aci = float(90)
    RadyanAci = Aci * 2 * math.pi / 360
    x2 = 0
    y2 = 0
    x0 = ResimGenisligi // 2
    y0 = ResimYuksekligi // 2

    for x1 in range(ResimGenisligi):
        for y1 in range(ResimYuksekligi):
            OkunanRenk = image.getpixel((x1, y1))
            Delta = (x1 - x0) * math.sin(RadyanAci) - (y1 - y0) * math.cos(RadyanAci)
            x2 = int(x1 + 2 * Delta * (-math.sin(RadyanAci)))
            y2 = int(y1 + 2 * Delta * (math.cos(RadyanAci)))
            if x2 > 0 and x2 < ResimGenisligi and y2 > 0 and y2 < ResimYuksekligi:
                CikisResmi.putpixel((x2, y2), OkunanRenk)  # Use CikisResmi after creating it

    return CikisResmi

#
## oteleme filter
##
def apply_oteleme(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Input image must be a PIL Image object.")

    ResimGenisligi = image.width
    ResimYuksekligi = image.height
    CikisResmi = Image.new("RGB", (ResimGenisligi, ResimYuksekligi))
    Tx = 100
    Ty = 50
    for x1 in range(ResimGenisligi):
        for y1 in range(ResimYuksekligi):
            OkunanRenk = image.getpixel((x1, y1))
            x2 = x1 + Tx
            y2 = y1 + Ty
            if 0 <= x2 < ResimGenisligi and 0 <= y2 < ResimYuksekligi:
                CikisResmi.putpixel((x2, y2), OkunanRenk)
    return CikisResmi

#
## Dondurme filter
#

def apply_dondurme(image, angle):
    if not isinstance(image, Image.Image):
        raise TypeError("Input image must be a PIL Image object.")

    image_width = image.width
    image_height = image.height
    output_image = Image.new("RGB", (image_width, image_height))

    radian_angle = angle * 2 * math.pi / 360  # Convert angle to radians
    center_x = image_width // 2
    center_y = image_height // 2

    for x1 in range(image_width):
        for y1 in range(image_height):
            pixel_color = image.getpixel((x1, y1))

            # Rotate coordinates
            x2 = int(math.cos(radian_angle) * (x1 - center_x) - math.sin(radian_angle) * (y1 - center_y) + center_x)
            y2 = int(math.sin(radian_angle) * (x1 - center_x) + math.cos(radian_angle) * (y1 - center_y) + center_y)

            # be sure rotated coordinates are within image bounds
            if 0 <= x2 < image_width and 0 <= y2 < image_height:
                output_image.putpixel((x2, y2), pixel_color)

    return output_image

#
## Ortalama
#

def apply_average_filter(image, filter_size):
    if filter_size == 0:
        raise ValueError("Filter size cannot be 0")

    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    half_filter_size = (filter_size - 1) // 2

    for x in range(max(half_filter_size, 0), width - half_filter_size):  # Adjust lower bounds
        for y in range(max(half_filter_size, 0), height - half_filter_size):  # Adjust lower bounds
            total_r, total_g, total_b = 0, 0, 0

            for i in range(-half_filter_size, half_filter_size + 1):
                for j in range(-half_filter_size, half_filter_size + 1):
                    pixel = image.getpixel((x + i, y + j))
                    total_r += pixel[0]
                    total_g += pixel[1]
                    total_b += pixel[2]

            average_r = total_r // (filter_size * filter_size)
            average_g = total_g // (filter_size * filter_size)
            average_b = total_b // (filter_size * filter_size)

            output_image.putpixel((x, y), (average_r, average_g, average_b))

    return output_image

#
## red filter
#

def adjust_red_channel(image, red_adjustment):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))

            r = pixel[0] + red_adjustment
            g = pixel[1]
            b = pixel[2]

            g = max(0, min(r, 255))

            output_pixel = (r, g, b)
            output_image.putpixel((x, y), output_pixel)

    return output_image
#
## green filter
#
def adjust_green_channel(image, green_adjustment):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))

            r = pixel[0]
            g = pixel[1]  + green_adjustment
            b = pixel[2]  # Keep blue unchanged for now

            b = max(0, min(r, 255))  # Clamp to valid range

            output_pixel = (r, g, b)
            output_image.putpixel((x, y), output_pixel)

    return output_image

#
## blue filter
#
def adjust_blue_channel(image, blue_adjustment):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))

            r = pixel[0]
            g = pixel[1]
            b = pixel[2]  + blue_adjustment

            b = max(0, min(r, 255))  # Clamp to valid range

            output_pixel = (r, g, b)
            output_image.putpixel((x, y), output_pixel)

    return output_image

#
## contrast filter
#

def adjust_contrast(image, contrast_level):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))
    contrast_factor = (259 * (contrast_level + 255)) / (255 * (259 - contrast_level))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            r = int((contrast_factor * (pixel[0] - 128)) + 128)
            g = int((contrast_factor * (pixel[1] - 128)) + 128)
            b = int((contrast_factor * (pixel[2] - 128)) + 128)

            if r > 255:
                r = 255
            if g > 255:
                g = 255
            if b > 255:
                b = 255
            if r < 0:
                r = 0
            if g < 0:
                g = 0
            if b < 0:
                b = 0

            output_image.putpixel((x, y), (r, g, b))

    return output_image

#
## parlaklik filter
#
def adjust_parlaklik(image, parlaklik_level):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            r = pixel[0] + parlaklik_level
            g = pixel[1] + parlaklik_level
            b = pixel[2] + parlaklik_level

            if r > 255:
                r = 255
            if g > 255:
                g = 255
            if b > 255:
                b = 255
            if r < 0:
                r = 0
            if g < 0:
                g = 0
            if b < 0:
                b = 0

            output_image.putpixel((x, y), (r, g, b))

    return output_image

#
## esikleme
#
def apply_esikleme(image, esikleme_value):
    width, height = image.size
    output_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if pixel[0] >= esikleme_value:
                r = 255
            else:
                r = 0
            if pixel[1] >= esikleme_value:
                g = 255
            else:
                g = 0
            if pixel[2] >= esikleme_value:
                b = 255
            else:
                b = 0

            output_image.putpixel((x, y), (r, g, b))

    return output_image

#
## Yayma filter
#

def apply_yayma(image, structuring_element):
    image_width = image.width
    image_height = image.height
    output_image = Image.new("RGB", (image_width, image_height))

    center_x = structuring_element.shape[0] // 2
    center_y = structuring_element.shape[1] // 2

    for x in range(image_width):
        for y in range(image_height):
            max_value = (1, 1, 1)  # Initialize as a tuple for RGB comparison
            for i in range(structuring_element.shape[0]):
                for j in range(structuring_element.shape[1]):
                    offset_x = x + i - center_x
                    offset_y = y + j - center_y
                    if 0 <= offset_x < image_width and 0 <= offset_y < image_height:
                        current_value = image.getpixel((offset_x, offset_y))
                        max_value = tuple(max(a, b) for a, b in zip(max_value, current_value))  # Compare each channel
            output_image.putpixel((x, y), max_value)

    return output_image


