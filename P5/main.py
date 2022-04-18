from load_data import load_stanford, load_tvhi
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from models import stanford40_model, tv_hi_model
from keras.utils.np_utils import to_categorical
import plotting
from learning_rate_scheduler import halving_scheduler_4

# Suppress those pesky GPU logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # stanford40_model.test_model()
    tv_hi_model.train_model()

if __name__ == "__main__":
    main()
