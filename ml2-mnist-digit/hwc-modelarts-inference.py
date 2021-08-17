from PIL import Image
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from model_service.tfserving_model_service import TfServingBaseService


class MnistService(TfServingBaseService):

    # Match the model input with the user's HTTPS API input during preprocessing.
    # The model input corresponding to the preceding training part is {"images":<array>}.
    def _preprocess(self, data):

        preprocessed_data = {}
        images = []
        # Iterate the input data.
        for k, v in data.items():
            for file_name, file_content in v.items():
                # load the image
                img = load_img(file_content, color_mode="grayscale", target_size=(28, 28))
                # convert to array
                img = img_to_array(img)
                # reshape into a single sample with 1 channel
                img = img.reshape(1, 28, 28, 1)
                # prepare pixel data
                img = img.astype('float32')
                img = img / 255.0
                images.append(img)

        # Return the numpy array.
        images = np.array(images, dtype=np.float32)
        # Perform batch processing on multiple input samples and ensure that
        # the shape is the same as that inputted during training.
        images.resize((len(data), 784))
        preprocessed_data['images'] = images
        return preprocessed_data

    # Processing logic of the inference for invoking the parent class.

    # The output corresponding to model saving in the preceding training part is {"scores":<array>}.
    # Postprocess the HTTPS output.
    def _postprocess(self, data):
        infer_output = {"mnist_result": []}
        # Iterate the model output.
        for output_name, results in data.items():
            for result in results:
                infer_output["mnist_result"].append(result.index(max(result)))
        return infer_output
