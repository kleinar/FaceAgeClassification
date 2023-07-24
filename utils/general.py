import numpy as np
import cv2

def visualize_image(img:np.array, face_coord:tuple, age:float):
    '''

    :param img: numpy array format image
    :param face_coord: (xmin, ymin,xmax, ymax)
    :param age: human age
    :return: visualize image with rectangles araund face and put text age of human

    '''
    start_point = (face_coord[0], face_coord[1])
    end_point = (face_coord[2], face_coord[3])
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    fontScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    img = cv2.rectangle(img, start_point, end_point, color, thickness)
    img = cv2.putText(img, str(age), start_point, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    return img