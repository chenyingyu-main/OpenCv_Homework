import keras
from tensorflow.python.keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa

###########################################################################################
# link of google colab:
# https://colab.research.google.com/drive/1q2stBG6YGKVQDdmOJnBzI11pB1-0TwCm?usp=share_link
#
###########################################################################################


# 資料路徑
DATASET_PATH  = '/content/drive/MyDrive/OpenCv/colab_Dataset_OpenCvDl_Hw2_Q5/Dataset_OpenCvDl_Hw2_Q5'

# 影像大小
IMAGE_SIZE = (224, 224)

# 影像類別數
NUM_CLASSES = 2

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 8

# 凍結網路層數
FREEZE_LAYERS = 2

# Epoch 數
NUM_EPOCHS = 5

# 模型輸出儲存的檔案
WEIGHTS_FINAL = '/content/drive/MyDrive/OpenCv/model-resnet50-catdog.h5'

# 透過 data augmentation 產生訓練與驗證用的影像資料
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/training_dataset',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/validation_dataset',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning

# 1st loss function
# ## cats: 5,413, dogs: 10,789 => a = 0.334
loss_function = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.334, gamma=1.0)

# 2nd loss function
# loss_function = tf.keras.losses.BinaryCrossentropy()

optimizer= Adam(learning_rate=8e-5)

net_final.compile(optimizer=optimizer,
                  loss=loss_function, metrics=['accuracy'])

# 輸出整個網路結構
print(net_final.summary())

# 訓練模型
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

# 儲存訓練好的模型
net_final.save(WEIGHTS_FINAL)