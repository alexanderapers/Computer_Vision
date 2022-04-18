

# Load both models
# Load both model weights
# Set all model weights to trainable=false

# Take the output of both model's flatten layers, and then combine those.
# It's also an option to keep a pretrained dense layer, since this time we're working with the same classes.

# keras has a concatenate/merge api layer thingie. Look into that.

# Then find a way to change our input to fit the new 2-stream model.

# train until the graphs look nice.
# test and record test score.
# Make sure training accuracy is above 30% before we choose a testing candidate