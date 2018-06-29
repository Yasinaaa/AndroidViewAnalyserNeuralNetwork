import numpy as np
from keras.preprocessing import image
from keras.models import load_model

test_image = image.load_img('dataset/single_prediction/test4.PNG', target_size = (24, 24))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
classifier = load_model("android.h5")
result = classifier.predict_proba(test_image)
print(result[0][0])
if result[0][0] == 1.0:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)