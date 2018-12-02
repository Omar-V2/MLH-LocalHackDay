from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation


img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)

img = load_img('data/train/five/1.jpg')
array_img = img_to_array(img)
print(array_img.shape)

train_dir = 'data/train'
num_train_samples = 500
epochs = 10
batch_size = 16




# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(5))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train_datagen = ImageDataGenerator(
#     rescale=1/255.,
#     validation_split=0.1
# )

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical'
# )

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=num_train_samples // batch_size,
#     epochs=epochs
# )

# model.save('trained.h5')