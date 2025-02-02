import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


loaded_model = load_model('classification_model.h5')
def preprocess_img(img):
    image = Image.open(img).resize((32,32))
    image = np.array(image)/255.0
    return np.expand_dims(image, axis=0)

img = '/Users/dany/Desktop/DS/2560px-A-Cat.jpg'
imgg = preprocess_img(img)
pred = loaded_model.predict(imgg)
pred_class = np.argmax(pred)
print(pred_class)