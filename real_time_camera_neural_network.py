import cv2
import numpy as np

from keras.models import load_model

model_RMSprop = load_model('../learnNN/NNLearning/dices_cnn/model2rms_continue-08-0.1021.h5')
model_Adam = load_model('../learnNN/NNLearning/dices_cnn/model3adam-25-0.1604.h5')


def capture():
    cap = cv2.VideoCapture(1)
    factor = 0.534
    window_name='Neural Network Recognition'
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        second_frame = frame.copy()
        y = int(200 * factor)
        x = int(260 * factor)
        size = int(120 * factor)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cropped_frame = frame[x:x+size, y:y+size]
        cv2.rectangle(second_frame, (x, y), (x+size, y+size), (255, 255, 255), 2)

        validation_frame = cropped_frame.reshape(1, 1, 64, 64).astype('float32')
        validation_frame /= 255
        prediction_rms = model_RMSprop.predict(validation_frame)
        prediction_adam = model_Adam.predict(validation_frame)

        results = '{} {}'.format(np.argmax(prediction_rms) + 1, np.argmax(prediction_adam) + 1)
        result_rms = 'RMS -> {}'.format(np.argmax(prediction_rms) + 1)
        result_adam = 'Adam -> {}'.format(np.argmax(prediction_adam) + 1)
        probability_rms = 'Probability RMS: {}'.format(np.amax(prediction_rms))
        probability_adam = 'Probability Adam: {}'.format(np.amax(prediction_adam))
        cv2.putText(second_frame, result_rms, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        cv2.putText(second_frame, probability_rms, (0, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        cv2.putText(second_frame, result_adam, (0, 215), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        cv2.putText(second_frame, probability_adam, (0, 245), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        cv2.putText(cropped_frame, results, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        cv2.imshow(window_name, cropped_frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture()

