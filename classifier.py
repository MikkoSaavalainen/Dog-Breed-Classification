import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def main():
    
    data = np.load("Processed/data.npy")
    labels = np.load("Processed/labels.npy")

    x_train, x_test, y_train, y_test = train_test_split( data, labels, 
                                                         test_size = 0.25,
                                                         random_state = 42)
    
    y_train = np_utils.to_categorical(y_train, 120)
    y_test = np_utils.to_categorical(y_test, 120)
    
#    x_train = x_train[..., np.newaxis]
#    x_test = x_test[..., np.newaxis]
    
    model = make_model()
    
    model.fit(x_train, y_train, epochs=10, batch_size=100, 
              validation_data = [x_test, y_test])
#    trained_model = train_model(model, x_train, y_train)
#    results = test_model(trained_model)
    
def make_model():
    
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
    from keras.optimizers import Adam
    
    model = Sequential()
    
    model.add( Conv2D( filters = 32, kernel_size = (3,3), padding = 'same',
                       input_shape = (64, 64, 3)))
    model.add( MaxPool2D(2,2))
    
    model.add( Conv2D( filters = 32, kernel_size = (3,3), padding = 'same'))
    model.add( MaxPool2D(2,2))
    
    model.add( Flatten() )
    
    model.add( Dense(120, activation='softmax') )
    model.add( Dropout(0.5) )
    
    model.summary()
    
    optimizer = Adam()
    
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model
    
#def train_model():
#    
#    
#    
#def test_model():
#    
#    
if __name__ == '__main__':
    main()