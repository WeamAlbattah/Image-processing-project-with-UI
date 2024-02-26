
import numpy as np
import self
from PIL import Image


#
## perspective filter
#
# def perspective_correction(src_points, dst_points, input_image):
#
#     # Combine source and destination points into a matrix
#     # pts = np.concatenate((src_points, dst_points), axis=1)
#     print("Shape of src_points:", src_points.shape)
#     print("Shape of dst_points:", dst_points.shape)
#     print("Shape of np.ones((8, 1)):", np.ones((8, 1)).shape)
#     # pts = np.concatenate((src_points, dst_points, np.ones((8, 1))), axis=1)
#     # pts = np.concatenate((src_points, dst_points, np.ones((4, 1))), axis=1)
#
#     ones_column = np.ones((len(src_points), 1))
#     pts = np.concatenate((src_points, dst_points, ones_column), axis=1)
#
#     z_coordinates = np.ones((len(src_points) + len(dst_points), 1))
#
#     A = np.hstack((pts[:, :2], z_coordinates, np.ones((8, 1))))
#     b = z_coordinates
#
#     M = np.linalg.lstsq(A, b, rcond=None)[0]
#
#     pts[:, :2] = np.zeros((8, 8))
#     for i in range(4):
#         pts[:, :2][i, :2] = src_points[i]
#         pts[:, :2][i + 4, 2:] = dst_points[i]
#         pts[:, :2][i, 8 - 1] = 1  # Set the last element of each row to 1
#
#     pts[:, 2:] = np.ones((8, 1))
#
#     M = np.linalg.lstsq(pts[:, :2], pts[:, 2:], rcond=None)[0]
#
#
#     # Calculate the transformation matrix using np.linalg.lstsq
#     # M = np.linalg.lstsq(pts[:, :2], pts[:, 2:], rcond=None)[0]
#     # Assuming A and b are already constructed correctly
#     M, residuals, rank, s = np.linalg.lstsq(pts[:, :2], pts[:, 2:], rcond=None)
#
#     print("Transformation Matrix:")
#     print(M)  # Print the calculated transformation matrix
#
#     print("Residuals:")
#     print(residuals)  # Print the residuals of the least-squares solution
#
#     print("Rank of A:")
#     print(rank)  # Print the rank of the matrix A
#
#     print("Singular Values of A:")
#     print(s)  # Print the singular values of A
#
#     print("Shape of M:")
#     print(M.shape)  # Ensure M is a 3x3 matrix
#
#     print("Values of M:")
#     print(M)  # Examine the values of M for any inconsistencies
#
#     print("Shape of A:")
#     print(pts[:, :2].shape)
#
#     print("Shape of b:")
#     print(pts[:, 2:].shape)
#
#
#
#
#
#     # Get image dimensions
#     img_width, img_height = input_image.size
#     # output_image = Image.new("RGB", (img_width, img_height))
#     output_image = Image.new("RGB", (img_width, img_height), color="black")
#
#
#     # for x in range(img_width):
#     #     for y in range(img_height):
#     #         r, g, b = input_image.getpixel((x, y))
#     #
#     #         # Apply the transformation matrix to the current pixel coordinates
#     #         transformed_point = M.dot(np.array([x, y, 1]))
#     #         X, Y = transformed_point  # Assign all elements of transformed_point
#     #         # X, Y = X / Z, Y / Z  # Now you can divide by Z
#
#     for x in range(img_width):
#         for y in range(img_height):
#             r, g, b = input_image.getpixel((x, y))
#
#             # Apply the transformation matrix and divide by homogeneous coordinate
#             transformed_point = M.dot(np.array([x, y, 1]))
#             X, Y, Z = transformed_point
#             X, Y = X / Z, Y / Z
#
#
#             # Normalize coordinates to image bounds and perform nearest-neighbor assignment
#             if 0 < X < img_width and 0 < Y < img_height:
#                 output_image.putpixel((int(X), int(Y)), (r, g, b))
#
#     return output_image




# def perspective_correction(src_points, dst_points, input_image):
#
#     z_coordinates = np.ones((len(src_points) + len(dst_points), 1))
#
#     # Construct A and b correctly, ensuring consistent shapes
#     # A = np.hstack((src_points, dst_points, z_coordinates, np.ones((4, 1))))
#     pts = np.concatenate((src_points, dst_points), axis=0)
#     print("Shape of pts before slicing:", pts.shape)
#     A = np.hstack((pts, z_coordinates, np.ones((8, 1))))
#     b = z_coordinates
#
#     # Calculate the transformation matrix M
#     M = np.linalg.lstsq(A, b, rcond=None)[0]
#     M = M.reshape(3, 3)
#
#     print("Transformation Matrix:")
#     print(M)
#
#     # Get image dimensions
#     img_width, img_height = input_image.size
#     output_image = Image.new("RGB", (img_width, img_height), color="black")
#
#     for x in range(img_width):
#         for y in range(img_height):
#             r, g, b = input_image.getpixel((x, y))
#
#             # Apply the transformation matrix and divide by homogeneous coordinate
#             transformed_point = M.dot(np.array([x, y, 1]))
#             X, Y, Z = transformed_point
#             X, Y = X / Z, Y / Z
#
#             # Normalize coordinates to image bounds and perform nearest-neighbor assignment
#             if 0 < X < img_width and 0 < Y < img_height:
#                 output_image.putpixel((int(X), int(Y)), (r, g, b))
#
#     return output_image



def perspective_correction(src_points, dst_points, input_image):

    z_coordinates = np.ones((len(src_points) + len(dst_points), 1))

    # Construct A and b correctly, ensuring consistent shapes

    pts = np.concatenate((src_points, dst_points), axis=0)
    A = np.hstack((pts, z_coordinates, np.ones((8, 1)), src_points.reshape(-1, 2), dst_points.reshape(-1, 2)))
    b = z_coordinates
    print("Shape of A:", A.shape)
    print("Shape of b:", b.shape)
    M = np.linalg.lstsq(A, b, rcond=None)[0]

    print("Shape of M before reshaping:", M.shape)

    print("pts before reshaping:", pts)
    print("src_points:", src_points)
    print("dst_points:", dst_points)
    print("pts:", pts)


    M = M.reshape(3, 3)  # Reshape M to 3x3 for correct multiplication

    print("Transformation Matrix:")
    print(M)

    # Get image dimensions
    img_width, img_height = input_image.size
    output_image = Image.new("RGB", (img_width, img_height), color="black")

    for x in range(img_width):
        for y in range(img_height):
            r, g, b = input_image.getpixel((x, y))

            # Apply the transformation matrix and perspective division
            transformed_point = M.dot(np.array([x, y, 1]))
            X, Y, Z = transformed_point
            X, Y = X / Z, Y / Z

            # Normalize coordinates and perform nearest-neighbor assignment
            if 0 < X < img_width and 0 < Y < img_height:
                output_image.putpixel((int(X), int(Y)), (r, g, b))


    return output_image
