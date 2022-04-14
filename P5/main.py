import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import load_data
import tensorflow as tf
from matplotlib import pyplot as plt

from models import stanford40_model
from keras.utils.np_utils import to_categorical

def main():
    # gathers the two datasets file names and class labels
    # they are already split into train, validation and test using stratification
    # train, validation and test variables are tuples of lists (file_names, label_names)
    # classes are the unique names of the classes that are used


    # get the actual list of files for train, validation and test from stanford dataset from the file names
    s40_train, s40_val, s40_test, s40_classes = load_data.load_stanford()
    # print(s40_train)[0]

    model = stanford40_model.get_model()
    model.fit(s40_train,
        validation_data=s40_val, batch_size=32)

    # load_data.load_tvhi(TVHI_train[0], TVHI_validation[0], TVHI_test[0])

    

if __name__ == "__main__":
    main()
