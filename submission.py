# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from tensorflow.keras.utils import load_img, img_to_array

# Import helper functions from utils.py
import utils

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        self.combinations = [[3,2,1,0],[3,2,0,1],[3,1,2,0],[3,0,2,1],[3,1,0,2],[3,0,1,2],[2,3,1,0],[2,3,0,1],[1,3,2,0],[0,3,2,1],[1,3,0,2],[0,3,1,2],[2,1,3,0],[2,0,3,1],[1,2,3,0],[0,2,3,1],[1,0,3,2],[0,1,3,2],[2,1,0,3],[2,0,1,3],[1,2,0,3],[0,2,1,3],[1,0,2,3],[0,1,2,3]]

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path:
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`

        OUTPUT:
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """
        # Load the image
        img = load_img(img_path, target_size=(128, 128))

        # Converts the image to a 3D numpy array (128x128x3)
        img_array = img_to_array(img)
        img_array /= 255

        img_0 = img_array[0:64, 0:64]
        img_1 = img_array[0:64, 64:]
        img_2 = img_array[64:, 0:64]
        img_3 = img_array[64:, 64:]
        img_split = [img_0, img_1, img_2, img_3]

        comb_diff = {}
        for comb in self.combinations:
            imgnew_0 = img_split[comb[0]]
            imgnew_1 = img_split[comb[1]]
            imgnew_2 = img_split[comb[2]]
            imgnew_3 = img_split[comb[3]]

            img_01_diff = imgnew_0[:, -1] - imgnew_1[:, 0]
            img_02_diff = imgnew_0[-1, :] - imgnew_2[0, :]
            img_23_diff = imgnew_2[:, -1] - imgnew_3[:, 0]
            img_13_diff = imgnew_1[-1, :] - imgnew_3[0, :]

            diff_absol1 = np.absolute(img_01_diff)
            diff_absol2 = np.absolute(img_02_diff)
            diff_absol3 = np.absolute(img_23_diff)
            diff_absol4 = np.absolute(img_13_diff)

            diff_absol_total = diff_absol1.sum() + diff_absol2.sum() + diff_absol3.sum() + diff_absol4.sum()

            comb_str = ""
            for x in comb:
                comb_str += str(x)

            comb_diff[comb_str] = diff_absol_total

        return min(comb_diff, key = comb_diff.get)[::-1]

# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':

    for img_name in glob('example_images/*'):
        # Open an example image using the PIL library
        example_image = Image.open(img_name)

        # Use instance of the Predictor class to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)
        # Example images are all shuffled in the "3120" order
        print(prediction)

        # Visualize the image
        pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        # Example images are all shuffled in the "3120" order
        final_image = Image.fromarray(np.vstack((np.hstack((pieces[3],pieces[1])),np.hstack((pieces[2],pieces[0])))))
        final_image.show()
