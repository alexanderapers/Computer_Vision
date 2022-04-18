import os
from models import stanford40_model, tv_hi_model


# Suppress those pesky GPU logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # tv_hi_model.train_model()
    tv_hi_model.test_model()

if __name__ == "__main__":
    main()
