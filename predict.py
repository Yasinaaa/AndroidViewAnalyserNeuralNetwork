import numpy as np
from keras.preprocessing import image
from keras.models import load_model

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
classifier = load_model("cats_dogs.h5")
result = classifier.predict(test_image)
print(result[0][0])
if result[0][0] == 1.0:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)