
from tensorflow import keras
from models import stanford40_model
from keras import backend as K
from keras import layers, Model

def get_model():

    model = stanford40_model.get_model()
    model.load_weights("training/stanford40-epoch0011").expect_partial()


    for layer in model.layers:
        layer.trainable = False
    
    x = model.layers[-2].output
    x = layers.Dense(160)(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(4)(x)

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model = Model(
        inputs = model.input,
        outputs = predictions)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])


    return model


