import cv2
import numpy as np

from keras.models import load_model
"""
Presentation of learned model ability with properly recognizing dots on dices
inside 106x79 pixels area, marked with rectangle
"""
model = load_model('models/simple_NN_106x79-20-0.7956.h5')


def check_for_break():
    k = cv2.waitKey(1)
    if k % 256 == 27:
        return True


def get_dimensions(cap):
    width = cap.get(3)
    height = cap.get(4)
    return width, height


def frame_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def generate_result_probability(prediction):
    result = 'Result:{}'.format(np.argmax(prediction) + 1)
    probability = 'Probability: {}'.format(np.amax(prediction))
    return result, probability


def draw_borders(second_frame, x, y, x_size, y_size):
    cv2.rectangle(second_frame, (x, y), (x + x_size, y + y_size), (255, 255, 255), 2)


def add_text_on_frame(second_frame, text, y_axis_offset):
    cv2.putText(second_frame, text, (0, y_axis_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)


def capture():
    cap = cv2.VideoCapture(1)
    width, height = get_dimensions(cap)
    print(width, height)
    factor = 0.25
    window_name='Neural Network Recognition'
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

        second_frame = frame.copy()
        y = int(int(height/2 - 90) * factor)
        x = int(int(width/2 - 110) * factor)
        x_size = 106
        y_size = 79
        frame = frame_to_gray(frame)

        cropped_frame = frame[y:y+y_size, x:x+x_size]
        draw_borders(second_frame, x, y, x_size, y_size)

        validation_frame = cropped_frame.reshape(1, x_size, y_size, 1).astype('float32')
        validation_frame /= 255
        prediction = model.predict(validation_frame)

        result, probability = generate_result_probability(prediction)

        add_text_on_frame(second_frame, result, 25)
        add_text_on_frame(second_frame, probability, 55)

        cv2.imshow(window_name, second_frame)

        if check_for_break():
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture()

