from keras import Model, layers, Input

# Input size
INPUT_SHAPE = (28,28,1)

class BaselineModel(Model):
    def __init__(self):
        super().__init__()
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10)



    def call(self, inputs):
        outputs = []
        for input in inputs:
            x = self.flatten(input)
            x = self.dense1(x)
            x = self.dense2(x)
            outputs.append(x)
        return outputs
