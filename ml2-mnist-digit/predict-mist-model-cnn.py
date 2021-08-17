# make a prediction for a new image.
import numpy
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# load and prepare the image


def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run_example():
    # load model
    model = load_model('model/model-cnn-mnist.h5')

    # load the image
    img = load_image('sample/preview2.png')
    # predict the class
    digit = model.predict_classes(img)
    print("predict ", digit)

    digit_detail = model.predict(img)
    labels_result = digit_detail.argmax(axis=-1)
    print(labels_result)

# entry point, run the example
run_example()
