# Perceptron

# AND 게이트
import numpy as np

def AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.8
  tmp = w1*x1 + w2*x2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
# 0
# 0
# 0
# 1



# 손글씨 숫자 데이터 인식하는 딥러닝 모델
import tensorflow as tf
from tensorflow import keras

tf.__version__
# 2.8.2

keras.__version__
# 2.8.0


# pip install keras==2.2.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)
# x_train shape (60000, 28, 28)
# y_train shape (60000,)
# x_test shape (10000, 28, 28)
# y_test shape (10000,)



x_train[0]
# array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
#          18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
#         253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
#         253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
#         253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
#         205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
#          90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
#         190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
#         253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
#         241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#          81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
#         148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
#         253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
#         253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
#         195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
#          11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0]], dtype=uint8)




# 첫 번째 테스트 데이터 확인
y_train[0]
# 5

# 딥러닝 모델이 원하는 형태로 데이터를 전처리

# 벡터화
X_train = x_train.reshape(60000,784)
X_test = x_test.reshape(10000,784)

# # 부동소주점화
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 정규화
X_train/= 255
X_test /= 255



# 레이블 데이터 원-핫 인코딩 처리
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)
print('Y Training matrix shape', Y_train.shape)
print('Y Testing matrix shape', Y_test.shape)
# Y Training matrix shape (60000, 10)
# Y Testing matrix shape (10000, 10)




Y_train[0]
# array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)



# 모델 생성 - 순전파
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
Model: "sequential_2"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense_3 (Dense)             (None, 512)               401920    
                                                                 
#  activation_3 (Activation)   (None, 512)               0         
                                                                 
#  dense_4 (Dense)             (None, 256)               131328    
                                                                 
#  activation_4 (Activation)   (None, 256)               0         
                                                                 
#  dense_5 (Dense)             (None, 10)                2570      
                                                                 
#  activation_5 (Activation)   (None, 10)                0         
                                                                 
# =================================================================
# Total params: 535,818
# Trainable params: 535,818
# Non-trainable params: 0
# _________________________________________________________________




# 모델 생성 - 역전파
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습
model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1)
# Epoch 1/10
# 469/469 [==============================] - 6s 12ms/step - loss: 0.2301 - accuracy: 0.9319
# Epoch 2/10
# 469/469 [==============================] - 6s 12ms/step - loss: 0.0821 - accuracy: 0.9752
# Epoch 3/10
# 469/469 [==============================] - 6s 13ms/step - loss: 0.0526 - accuracy: 0.9837
# Epoch 4/10
# 469/469 [==============================] - 5s 12ms/step - loss: 0.0364 - accuracy: 0.9886
# Epoch 5/10
# 469/469 [==============================] - 6s 12ms/step - loss: 0.0279 - accuracy: 0.9910
# Epoch 6/10
# 469/469 [==============================] - 6s 12ms/step - loss: 0.0215 - accuracy: 0.9929
# Epoch 7/10
# 469/469 [==============================] - 6s 12ms/step - loss: 0.0170 - accuracy: 0.9942
# Epoch 8/10
# 469/469 [==============================] - 6s 12ms/step - loss: 0.0174 - accuracy: 0.9941
# Epoch 9/10
# 469/469 [==============================] - 6s 12ms/step - loss: 0.0123 - accuracy: 0.9960
# Epoch 10/10
# 469/469 [==============================] - 5s 11ms/step - loss: 0.0129 - accuracy: 0.9954
# <keras.callbacks.History at 0x7f8d7cff97d0>





# 새로운 데이터를 통해 모델의 현재 수준을 평가
# score = 
print('Test score:', score[0])





# Fashion MNIST 분류기

# 데이터 로딩
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()


# 데이터 살펴보기
X_train_full.shape
# (60000, 28, 28)



X_train_full.dtype
# dtype('uint8')


# 훈련데이터, 검증데이터 분리 - 정규화, 부동소수점형
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test_full = X_test_full / 255.0



import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


plt.imshow(X_train[0], cmap='binary')
plt.axis('off')
plt.show()
# '코트 사진 나옴'


y_train
# array([4, 0, 7, ..., 3, 0, 5], dtype=uint8)



# 출력될 클래스 이름 지정
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names[y_train[0]]
# Coat




# 데이터셋에 있는 이미지를 샘플로 확인
n_rows = 4
n_columns = 10

plt.figure(figsize=(n_columns * 1.2, n_rows * 1.2))

for row in range(n_rows):
  for col in range(n_columns):
    index = n_columns * row + col
    plt.subplot(n_rows, n_columns, index + 1)
    plt.imshow(X_train[index], cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.title(class_names[y_train[index]], fontsize=12)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
# 상품들 이름, 사진들 나옴




### ----- 다시

# 데이터 로딩
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()


# 출력될 클래스 이름 지정
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


X_train.shape
# (55000, 28, 28)

y_train
# array([4, 0, 7, ..., 3, 0, 5], dtype=uint8)


# 정규화
X_train = X_train / 255.0
y_train = y_train / 255.0


# 모델 생성
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
# Model: "sequential_6"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten_3 (Flatten)         (None, 784)               0         
                                                                 
#  dense_9 (Dense)             (None, 300)               235500    
                                                                 
#  dense_10 (Dense)            (None, 100)               30100     
                                                                 
#  dense_11 (Dense)            (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 266,610
# Trainable params: 266,610
# Non-trainable params: 0
# _________________________________________________________________



# ------------


keras.utils.plot_model(model, 'fashion_mnist_model.png', show_shapes=True)
# 모델 사진


# 모델 컴파일
model.compile(loss='sparse_categorical_crossentpy', optimizer='sgd',
              metrics=['accuracy'])

# '''
# # sparse_categorical_crossentry : 레이블이 정수하나로 이루어져 있고 (즉, 샘플마다
#   타깃 클래스 하나) => 베타적인 클래스 라면 유용

# # categorical_crossentry
#   : 샘플마다 클래스별 타깃 확률을 가지고 있다면 유용
#   : one-hot encoding
#   ex) 클래스 3 [0, 0, 1, 0, 0, 0, 0, 0]

# # 이진분류 : loss=sigmoid 유용 / binary-crossentry

# '''




# 에러 발생
# # 학습
# # history = model.fit(X_train, y_train, epochs=30, verbose=1)

# # history.params

# # history.history.keys()

# # 학습곡선
# # import pandas as pd

# # pd.DataFrame(history.history).plot(figsize=(8.5))
# # plt.grid(True)
# # plt.gca().set_ylim(0,1)
# # plt.show()

# # # 평가
# # model.evaluate(X_test, y_test)