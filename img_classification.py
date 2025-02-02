import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


(x_train, y_train),(x_test, y_test) = cifar10.load_data()
#database visualization
# print(x_train.shape)
# print(x_test.shape)
# print(len(set(y_train.flatten())))
# clases = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Sheep', 'Truck']
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(x_train[i])
#     plt.title(clases[y_train[i][0]])
#     plt.axis('off')
# plt.tight_layout()
# plt.show()


# Normalizind image data
x_train = x_train/255
x_test = x_test/255

# One-hot-encoder for categories
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#Split x_train to validation and  train dataset
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Creatin CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32,(3,3), activation = 'relu', input_shape =(32, 32, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dropout(0.5),
    Dense(10, activation = 'softmax')
])
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# print(model.summary())

#обучение модели и оценка ее производительности
# history = model.fit(
#     x_train, y_train, epochs = 10, batch_size = 64, validation_data = (x_val, y_val)
# )
#Оценка модели
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
# print(test_acc*100)

#visualization
# plt.plot(history.history['accuracy'], label = 'Accuracy')
# plt.plot(history.history['val_accuracy'], label = 'Valid accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# Настройка гиерпараметров модели
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoches, lr):
    if epoches < 5:
        return lr
    else:
        return lr*0.5
    
callback = LearningRateScheduler(scheduler)
optimizer = Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# history = model.fit(
#     x_train, y_train, epochs = 10, validation_data = (x_val, y_val), callbacks = [callback]
# )
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
# print(test_acc*100)


#Заупскаем модель с аугментированой бд. accuracy по сравнению с предидущей моделью упадет
# temp = ImageDataGenerator(
#     rotation_range = 20, #поворот на градусы
#     shear_range = 0.1, #сдвиг угла изображения от 0 до 1
#     zoom_range = 0.2, #зум изображения, где 1 - оригинальный размер
#     horizontal_flip = True, #mirroring
#     brightness_range = (0.5, 1.5),
#     width_shift_range = 0.2, #сдвиг по оси ширины влево/вправо
#     height_shift_range = 0.2, # same for height
#     fill_mode = 'nearest' #заполнение пробелов (см другие парам) 
#     )

# temp.fit(x_train)
# history = model.fit(temp.flow(x_train, y_train, batch_size = 64), epochs = 10, validation_data = (x_val, y_val)) 

#save model to .h5
# model.save('classification_model.h5')


