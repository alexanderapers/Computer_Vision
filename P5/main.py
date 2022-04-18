import os
from models import stanford40_model, tv_hi_model, tvhi_flows_model


# Suppress those pesky GPU logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # tvhi_flows_model.train_model()
    tvhi_flows_model.test_model()

if __name__ == "__main__":
    main()
