import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import np_utils

def main():
    
    data = np.load("Processed/data.npy")
    labels = np.load("Processed/labels.npy")

    x_train, x_test, y_train, y_test = train_test_split( data, labels, 
                                                         test_size = 0.25,
                                                         random_state = 51)
    
    y_train = np_utils.to_categorical(y_train, 120)
    y_test = np_utils.to_categorical(y_test, 120)
    
#    x_train = x_train[..., np.newaxis]
#    x_test = x_test[..., np.newaxis]
    
    model = make_model()
    
    # x_test and y_test are provided for the model training phase so that we
    # can see accuracies during training
    trained_model = train_model(model, x_train, y_train, x_test, y_test, 20, 100)
    results = test_model(trained_model, x_test)

    predictions = np.argmax(results, axis = 1)
    predictions = np_utils.to_categorical(predictions, 120)
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)

def make_model():
    
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
    from keras.layers import GlobalAveragePooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam
    
    model = Sequential()
    
    model.add( BatchNormalization( input_shape=(100, 100, 3) ) )
    model.add( Conv2D( filters = 32, kernel_size = (3,3), activation='relu',
                       kernel_initializer='he_normal'))
    model.add( MaxPool2D(2,2) )
    model.add( BatchNormalization() )
    
    model.add( Conv2D( filters = 64, kernel_size = (3,3), activation='relu',
                       kernel_initializer='he_normal'))
    model.add( MaxPool2D(2,2) )
    model.add( BatchNormalization() )
    
    model.add( Conv2D( filters = 128, kernel_size = (3,3), activation='relu',
                       kernel_initializer='he_normal'))
    model.add( MaxPool2D(2,2) )
    model.add( BatchNormalization() )
    
    model.add( Conv2D( filters = 256, kernel_size = (3,3), activation='relu',
                       kernel_initializer='he_normal'))
    model.add( MaxPool2D(2,2) )
    model.add( BatchNormalization() )
    
#    model.add( Flatten() )
#
#    model.add( Dense(256, activation='relu' ) )
#    model.add( Dropout(0.9) )
    
    model.add( GlobalAveragePooling2D() )
    
    model.add( Dense(120, activation='softmax') )
    
    model.summary()
    
    optimizer = Adam()
    
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model
    
def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_data = [x_test, y_test])
    
    return model
    
def test_model(model, x_test):
    results = model.predict(x_test)
    
    return results
    
if __name__ == '__main__':
    main()