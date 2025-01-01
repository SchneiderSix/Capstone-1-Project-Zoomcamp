import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    '/content/datasets/test',
    target_size=(350, 350),
    batch_size=32,
    shuffle=False
)

model = ks.models.load_model('xception_v1_26_0.980.keras')

model.evaluate(test_ds)

path = '/content/datasets/test/wildfire/-59.03238,51.85132.jpg'

img = ks.preprocessing.image.load_img(path, target_size=(350, 350))

x = np.array(img)
X = np.array([x])

X = preprocess_input(X)

pred = model.predict(X)

classes = [
    'nowildfire',
    'wildfire'
]

dict(zip(classes, pred[0]))
