import cv2
import numpy as np

from keras.models import load_model

model_RMSprop = load_model('models/model2rms_continue-08-0.1021.h5')
model_Adam = load_model('models/model3adam-25-0.1604.h5')
model_generator = load_model('models/comparison_gray-20-0.0402.h5')
model_AlexNet = load_model('models/dice_AlexNet-25-0.0644.h5')
model_API = load_model('models/model2API-25-0.0871.h5')


def capture():
    cap = cv2.VideoCapture(1)
    width = cap.get(3)
    height = cap.get(4)
    print(width, height)
    factor = 0.25
    window_name='Neural Network Recognition'
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        second_frame = frame.copy()
        y = int(int(height/2 - 100) * factor)
        x = int(int(width/2 - 130) * factor)
        size = 64
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(x, y, size, x+size, y+size)
        cropped_frame = frame[y:y+size, x:x+size]
        cv2.rectangle(second_frame, (x, y), (x+size, y+size), (255, 255, 255), 2)

        validation_frame = cropped_frame.reshape(1, 64, 64, 1).astype('float32')
        validation_frame_reversed = cropped_frame.reshape(1, 1, 64, 64).astype('float32')
        # print('Validation_frame ', validation_frame.shape)
        validation_frame /= 255
        prediction_rms = model_RMSprop.predict(validation_frame_reversed)
        prediction_adam = model_Adam.predict(validation_frame_reversed)
        prediction_gen = model_generator.predict(validation_frame)
        prediction_alex = model_AlexNet.predict(validation_frame)
        prediction_api = model_API.predict(validation_frame)

        result_rms = 'RMS:{}'.format(np.argmax(prediction_rms) + 1)
        result_adam = 'Adam:{}'.format(np.argmax(prediction_adam) + 1)
        result_gen = 'Gen:{}'.format(np.argmax(prediction_gen) + 1)
        result_alex = 'AlexNet:{}'.format(np.argmax(prediction_alex) + 1)
        result_api = 'API:{}'.format(np.argmax(prediction_api) + 1)

        probability_rms = 'Probability RMS: {}'.format(np.amax(prediction_rms))
        probability_adam = 'Probability Adam: {}'.format(np.amax(prediction_adam))

        cv2.putText(second_frame, result_rms, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        # cv2.putText(second_frame, probability_rms, (0, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)

        cv2.putText(second_frame, result_adam, (0, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        # cv2.putText(second_frame, probability_adam, (0, 175), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)

        cv2.putText(second_frame, result_gen, (0, 85), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)

        cv2.putText(second_frame, result_alex, (0, 115), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)

        cv2.putText(second_frame, result_api, (0, 145), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)

        # cv2.putText(cropped_frame, results, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, 8)
        cv2.imshow(window_name, second_frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture()

