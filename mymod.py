import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import VGG16
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os

my_model= VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
my_feautures_model= Model(inputs=my_model.input, outputs=my_model.layers[18].output)
my_feautures_model.trainable=False # Freezing VGG layers

train_datagen= ImageDataGenerator(rescale=1.0/255)
train_path="images/nfol/train"

train= train_datagen.flow_from_directory(
    train_path, target_size=(224,224), batch_size=32, class_mode=None)

# preparing data arrays
X , Y=[],[]
for img in train:
    lab= rgb2lab(img)
    X.append(lab[:,:,0])  # L channel as input
    Y.append(lab[:,:,1:]/128) # A & B channels as output

X,Y= np.array(X), np.array(Y)
X=X.reshape(X.shape+(1,))  # for reshaping for compatibility

# in this part we convert L channels to 3d input for VGG model

my_feautures_model=[]
for i, sample in enumerate(X):
    sample= gray2rgb(sample)
    sample= sample.reshape((1,224,224,3))
    prediction= my_feautures_model.predict(sample)
    my_feautures_model.append(prediction.reshape((7,7,512))) # tha last space from VGG
my_feautures_model=np.array(my_feautures_model)


# creating thr DECODER model for colorization
decoder = Sequential([
    Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 512)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(2, (3, 3), activation='tanh', padding='same'),
    UpSampling2D((2, 2))
])

decoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
decoder.summary()


# the training
decoder.fit(my_feautures_model, Y, epochs=100, batch_size=16, validation_split=0.1, verbose=1)
decoder.save('colorization_model_vgg16_tf.h5')

# lets predict on the test images
test_path = 'images/nfol/test/'
output_path = 'images/nfol/output/'
os.makedirs(output_path, exist_ok=True)
for idx,file in enumerate(os.listdir(test_path)):
    test_img= img_to_array(load_img(os.path.join(test_path,file)))
    test_img= resize(test_img,(224,224))/225.0
    lab=rgb2lab(test_img)
    L=lab[:,:,0]
    L_input= gray2rgb(L).reshape((1,224,224,3))

    # predicting features with VGG model
    my_output= my_feautures_model.predict(L_input)
    # predicting AB channels with the decoder
    ab_output=decoder.predict(my_output).reshape((224,224,2))* 128

    # mix the L & AB channels
    result_img = np.zeros((224, 224, 3))
    result_img[:, :, 0] = L
    result_img[:, :, 1:] = ab_output
    result_rgb = lab2rgb(result_img)


    # the saving part
    imsave(os.path.join(output_path, f"colorized_{idx}.jpg"),result_rgb)
    print(f"saved colorized image ::{output_path}colorized_{idx}.jpg")
