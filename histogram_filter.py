from email.mime import image

import self
from PyQt5.QtGui import qRed, qGreen, qBlue


# def generate_histogram(image, canvas, label, max_pixel_count):
#     pixel_array = []
#     average_color = 0
#     for x in range(image.width):  # Use the image's width and height
#         for y in range(image.height):
#             pixel = image.getpixel((x, y))  # Access pixels from the image
#             average_color = (pixel[0] + pixel[1] + pixel[2]) // 3
#             pixel_array.append(average_color)
#
#
#         pixel_counts = [0] * 256
#         for pixel in pixel_array:
#             pixel_counts[pixel] += 1
#
#         max_pixel_count = max(pixel_counts)
#
#         canvas.clear()  # Use the passed canvas object
#         canvas.addLine(10, 400, 10, 0)  # Adjust drawing code for your canvas
#         canvas.addLine(10, 400, 410, 400
#
#         for x in range(0, 256, 50):
#             self.canvas.addLine(10 + x, 400, 10 + x, 0, self.penRed)  # vertical lines
#
#         for x in range(256):
#             self.scene.addLine(10 + x, 400, 10 + x, 400 - (pixel_counts[x] / max_pixel_count) * 400, self.penBlack)  # histogram bars
#
#             max_pixel_count.setText("Max Pixel Count: " + str(max_pixel_count))

def generate_histogram(image, canvas, label, max_pixel_count_label, red_pen):
    pixel_array = []
    average_color = 0
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            average_color = (qRed(pixel[0]) + qGreen(pixel[1]) + qBlue(pixel[2])) // 3
            pixel_array.append(average_color)

    pixel_counts = [0] * 256
    for pixel in pixel_array:
        pixel_counts[pixel] += 1

    max_pixel_count = max(pixel_counts)

    canvas.clear()
    canvas.addLine(10, 400, 10, 0)  # Close the parenthesis
    canvas.addLine(10, 400, 410, 400)

    for x in range(0, 256, 50):
        canvas.addLine(10 + x, 400, 10 + x, 0, pen=red_pen)  # Pass the pen object

    for x in range(256):
        canvas.addLine(10 + x, 400, 10 + x, 400 - (pixel_counts[x] / max_pixel_count) * 400)

    max_pixel_count_label.setText("Max Pixel Count: " + str(max_pixel_count))  # Use the label object
