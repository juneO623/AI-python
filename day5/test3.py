# # [실습-퀴즈] 딥러닝 파트
# + 이번시간에는 Python을 활용한 AI 모델링에서 딥러닝에 대해 실습해 보겠습니다.
# + 여기서는 딥러닝 모델 DNN에 대해 코딩하여 모델 구축해 보겠습니다.
# + 이론보다 실습이 더 많은 시간과 노력이 투자 되어야 합니다.


# ## 실습목차
# +. 딥러닝 심층신경망(DNN) 모델 프로세스
#  - 데이터 가져오기
#  - 데이터 전처리
#  - Train, Test 데이터셋 분할
#  - 데이터 정규화
#  - DNN 딥러닝 모델


# #  
# # 1. 딥러닝 심층신경망(DNN) 모델 프로세스
# ① 라이브러리 임포트(import)  
# ② 데이터 가져오기(Loading the data)  
# ③ 탐색적 데이터 분석(Exploratory Data Analysis)  
# ④ 데이터 전처리(Data PreProcessing) : 데이터타입 변환, Null 데이터 처리, 누락데이터 처리, 
# 더미특성 생성, 특성 추출 (feature engineering) 등  
# ⑤ Train, Test  데이터셋 분할  
# ⑥ 데이터 정규화(Normalizing the Data)  
# ⑦ 모델 개발(Creating the Model)  
# ⑧ 모델 성능 평가



## ① 라이브러리 임포트
##### 필요 라이브러리 임포트

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ② 데이터 로드
##### <font color=blue> **[문제] 같은 폴더내에 있는 data_v1_save.csv 파일을 Pandas read_csv 함수를 이용하여 읽어 df 변수에 저장하세요.** </font>

# 읽어 들일 파일명 : data_v1_save.csv
# Pandas read_csv 함수 활용
# 결과 : df 저장


## ③ 데이터 분석
# 10컬럼, 9930 라인
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 7027 entries, 0 to 7026
# Data columns (total 17 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   gender            7027 non-null   object 
#  1   Partner           7027 non-null   object 
#  2   Dependents        7027 non-null   object 
#  3   tenure            7027 non-null   int64  
#  4   MultipleLines     7027 non-null   object 
#  5   InternetService   7027 non-null   object 
#  6   OnlineSecurity    7027 non-null   object 
#  7   OnlineBackup      7027 non-null   object 
#  8   TechSupport       7027 non-null   object 
#  9   StreamingTV       7027 non-null   object 
#  10  StreamingMovies   7027 non-null   object 
#  11  Contract          7027 non-null   object 
#  12  PaperlessBilling  7027 non-null   object 
#  13  PaymentMethod     7027 non-null   object 
#  14  MonthlyCharges    7027 non-null   float64
#  15  TotalCharges      7027 non-null   float64
#  16  Churn             7027 non-null   int64  
# dtypes: float64(2), int64(2), object(13)
# memory usage: 933.4+ KB




df.tail()
# 	gender	Partner	Dependents	tenure	MultipleLines	InternetService	OnlineSecurity	OnlineBackup	TechSupport	StreamingTV	StreamingMovies	Contract	PaperlessBilling	PaymentMethod	MonthlyCharges	TotalCharges	Churn
# 7022	Female	No	No	72	No	No	No internet service	No internet service	No internet service	No internet service	No internet service	Two year	Yes	Bank transfer (automatic)	21.15	1419.40	0
# 7023	Male	Yes	Yes	24	Yes	DSL	Yes	No	Yes	Yes	Yes	One year	Yes	Mailed check	84.80	1990.50	0
# 7024	Female	Yes	Yes	72	Yes	Fiber optic	No	Yes	No	Yes	Yes	One year	Yes	Credit card (automatic)	103.20	7362.90	0
# 7025	Female	Yes	Yes	11	No phone service	DSL	Yes	No	No	No	No	Month-to-month	Yes	Electronic check	29.60	346.45	0
# 7026	Male	Yes	No	4	Yes	Fiber optic	No	No	No	No	No	Month-to-month	Yes	Mailed check	74.40	306.60	1



# Churn 레이블 불균형 
df['Churn'].value_counts().plot(kind='bar')



# ## ④ 데이터 전처리
# + 모든 데이터값들은 숫자형으로 되어야 한다. 즉, Ojbect 타입을 모든 숫자형 변경 필요
# + Object 컬럼에 대해 Pandas get_dummies 함수 활용하여 One-Hot-Encoding



# Object 컬럼명 수집

cal_cols = df.select_dtypes('object').columns.values
cal_cols
# array(['gender', 'Partner', 'Dependents', 'MultipleLines',
#        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
#        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#        'PaymentMethod'], dtype=object)



##### <font color=blue> **[문제] Object 컬럼에 대해 One-Hot-Encoding 수행하고 그 결과를 df1 변수에 저장하세요.** </font>
# Pandas get_dummies() 함수 이용
# 원-핫-인코딩 결과를 df1 저장

# 19컬럼, 7814 라인
df1.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 7027 entries, 0 to 7026
# Data columns (total 40 columns):
#  #   Column                                   Non-Null Count  Dtype  
# ---  ------                                   --------------  -----  
#  0   tenure                                   7027 non-null   int64  
#  1   MonthlyCharges                           7027 non-null   float64
#  2   TotalCharges                             7027 non-null   float64
#  3   Churn                                    7027 non-null   int64  
#  4   gender_Female                            7027 non-null   uint8  
#  5   gender_Male                              7027 non-null   uint8  
#  6   Partner_No                               7027 non-null   uint8  
#  7   Partner_Yes                              7027 non-null   uint8  
#  8   Dependents_No                            7027 non-null   uint8  
#  9   Dependents_Yes                           7027 non-null   uint8  
#  10  MultipleLines_No                         7027 non-null   uint8  
#  11  MultipleLines_No phone service           7027 non-null   uint8  
#  12  MultipleLines_Yes                        7027 non-null   uint8  
#  13  InternetService_DSL                      7027 non-null   uint8  
#  14  InternetService_Fiber optic              7027 non-null   uint8  
#  15  InternetService_No                       7027 non-null   uint8  
#  16  OnlineSecurity_No                        7027 non-null   uint8  
#  17  OnlineSecurity_No internet service       7027 non-null   uint8  
#  18  OnlineSecurity_Yes                       7027 non-null   uint8  
#  19  OnlineBackup_No                          7027 non-null   uint8  
#  20  OnlineBackup_No internet service         7027 non-null   uint8  
#  21  OnlineBackup_Yes                         7027 non-null   uint8  
#  22  TechSupport_No                           7027 non-null   uint8  
#  23  TechSupport_No internet service          7027 non-null   uint8  
#  24  TechSupport_Yes                          7027 non-null   uint8  
#  25  StreamingTV_No                           7027 non-null   uint8  
#  26  StreamingTV_No internet service          7027 non-null   uint8  
#  27  StreamingTV_Yes                          7027 non-null   uint8  
#  28  StreamingMovies_No                       7027 non-null   uint8  
#  29  StreamingMovies_No internet service      7027 non-null   uint8  
#  30  StreamingMovies_Yes                      7027 non-null   uint8  
#  31  Contract_Month-to-month                  7027 non-null   uint8  
#  32  Contract_One year                        7027 non-null   uint8  
#  33  Contract_Two year                        7027 non-null   uint8  
#  34  PaperlessBilling_No                      7027 non-null   uint8  
#  35  PaperlessBilling_Yes                     7027 non-null   uint8  
#  36  PaymentMethod_Bank transfer (automatic)  7027 non-null   uint8  
#  37  PaymentMethod_Credit card (automatic)    7027 non-null   uint8  
#  38  PaymentMethod_Electronic check           7027 non-null   uint8  
#  39  PaymentMethod_Mailed check               7027 non-null   uint8  
# dtypes: float64(2), int64(2), uint8(36)
# memory usage: 466.8 KB





## ⑤ Train, Test  데이터셋 분할

from sklearn.model_selection import train_test_split

X = df1.drop('Churn', axis=1).values
y = df1['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y,
                                                    random_state=42)


X_train.shape
# (4918, 39)



y_train.shape
# (4918,)


## ⑥ 데이터 정규화/스케일링(Normalizing/Scaling)



# 숫자 분포 이루어진 컬럼 확인
# df1.tail()
# 	tenure	MonthlyCharges	TotalCharges	Churn	gender_Female	gender_Male	Partner_No	Partner_Yes	Dependents_No	Dependents_Yes	MultipleLines_No	MultipleLines_No phone service	MultipleLines_Yes	InternetService_DSL	InternetService_Fiber optic	InternetService_No	OnlineSecurity_No	OnlineSecurity_No internet service	OnlineSecurity_Yes	OnlineBackup_No	OnlineBackup_No internet service	OnlineBackup_Yes	TechSupport_No	TechSupport_No internet service	TechSupport_Yes	StreamingTV_No	StreamingTV_No internet service	StreamingTV_Yes	StreamingMovies_No	StreamingMovies_No internet service	StreamingMovies_Yes	Contract_Month-to-month	Contract_One year	Contract_Two year	PaperlessBilling_No	PaperlessBilling_Yes	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check
# 7022	72	21.15	1419.40	0	1	0	1	0	1	0	1	0	0	0	0	1	0	1	0	0	1	0	0	1	0	0	1	0	0	1	0	0	0	1	0	1	1	0	0	0
# 7023	24	84.80	1990.50	0	0	1	0	1	0	1	0	0	1	1	0	0	0	0	1	1	0	0	0	0	1	0	0	1	0	0	1	0	1	0	0	1	0	0	0	1
# 7024	72	103.20	7362.90	0	1	0	0	1	0	1	0	0	1	0	1	0	1	0	0	0	0	1	1	0	0	0	0	1	0	0	1	0	1	0	0	1	0	1	0	0
# 7025	11	29.60	346.45	0	1	0	0	1	0	1	0	1	0	1	0	0	0	0	1	1	0	0	1	0	0	1	0	0	1	0	0	1	0	0	0	1	0	0	1	0
# 7026	4	74.40	306.60	1	0	1	0	1	1	0	0	0	1	0	1	0	1	0	0	1	0	0	1	0	0	1	0	0	1	0	0	1	0	0	0	1	0	0	0	1




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



X_train[:2]
# array([[0.65277778, 0.56851021, 0.40877722, 1.        , 0.        ,
#         1.        , 0.        , 1.        , 0.        , 1.        ,
#         0.        , 0.        , 0.        , 1.        , 0.        ,
#         0.        , 0.        , 1.        , 1.        , 0.        ,
#         0.        , 1.        , 0.        , 0.        , 1.        ,
#         0.        , 0.        , 1.        , 0.        , 0.        ,
#         1.        , 0.        , 0.        , 1.        , 0.        ,
#         0.        , 1.        , 0.        , 0.        ],
#        [0.27777778, 0.00498256, 0.04008671, 1.        , 0.        ,
#         1.        , 0.        , 1.        , 0.        , 1.        ,
#         0.        , 0.        , 0.        , 0.        , 1.        ,
#         0.        , 1.        , 0.        , 0.        , 1.        ,
#         0.        , 0.        , 1.        , 0.        , 0.        ,
#         1.        , 0.        , 0.        , 1.        , 0.        ,
#         1.        , 0.        , 0.        , 0.        , 1.        ,
#         0.        , 1.        , 0.        , 0.        ]])




## ⑦ 딥러닝 심층신경망(DNN) 모델 구현
####  
### 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

tf.random.set_seed(100)


####  
### 하이퍼파라미터 설정 : batch_size, epochs
batch_size = 16
epochs = 20


####  
### 모델 입력(features) 갯수 확인
X_train.shape
# (4918, 39)


####  
### 모델 출력(label) 갯수 확인
y_train.shape



####  
### A. 이진분류 DNN모델 구성 

# hidden Layer
# [출처] https://subscription.packtpub.com/book/data/9781788995207/1/ch01lvl1sec03/deep-learning-intuition



##### <font color=blue> **[문제] 요구사항대로 Sequential 모델을 만들어 보세요.** </font>
# Sequential() 모델 정의 하고 model로 저장
# input layer는 input_shape=() 옵션을 사용한다.
# 39개 input layer
# unit 4개 hidden layer
# unit 3개 hidden layer 
# 1개 output layser : 이진분류




####  
### 모델 확인
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 4)                 160       
                                                                 
#  dense_1 (Dense)             (None, 3)                 15        
                                                                 
#  dense_2 (Dense)             (None, 1)                 4         
                                                                 
# =================================================================
# Total params: 179
# Trainable params: 179
# Non-trainable params: 0
# _________________________________________________________________




####  
### 모델 구성 -  과적합 방지
# dropout
# [출처] https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5



model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(39,)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))



####  
### 과적합 방지 모델 확인


model.summary()
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense_3 (Dense)             (None, 4)                 160       
                                                                 
#  dropout (Dropout)           (None, 4)                 0         
                                                                 
#  dense_4 (Dense)             (None, 3)                 15        
                                                                 
#  dropout_1 (Dropout)         (None, 3)                 0         
                                                                 
#  dense_5 (Dense)             (None, 1)                 4         
                                                                 
# =================================================================
# Total params: 179
# Trainable params: 179
# Non-trainable params: 0
# _________________________________________________________________




####  
### 모델 컴파일 – 이진 분류 모델
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy']) 

# - 모델 컴파일 – 다중 분류 모델 (Y값을 One-Hot-Encoding 한경우) <br>
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# - 모델 컴파일 – 다중 분류 모델  (Y값을 One-Hot-Encoding 하지 않은 경우) <br>
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# - 모델 컴파일 – 예측 모델
# model.compile(optimizer='adam', loss='mse')



####  
### 모델 학습



# ##### <font color=blue> **[문제] 요구사항대로 DNN 모델을 학습시키세요.** </font>
# + 모델 이름 : model
# + epoch : 10번
# + batch_size : 10번




# 앞쪽에서 정의된 모델 이름 : model
# Sequential 모델의 fit() 함수 사용
# @인자
### X, y : X_train, y_train
### validation_data=(X_test, y_test)
### epochs 10번
### batch_size 10번






# Epoch 1/10
# 492/492 [==============================] - 2s 3ms/step - loss: 0.6015 - accuracy: 0.7318 - val_loss: 0.5247 - val_accuracy: 0.7345
# Epoch 2/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5439 - accuracy: 0.7416 - val_loss: 0.4802 - val_accuracy: 0.7345
# Epoch 3/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5225 - accuracy: 0.7548 - val_loss: 0.4734 - val_accuracy: 0.7345
# Epoch 4/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5139 - accuracy: 0.7623 - val_loss: 0.4646 - val_accuracy: 0.7364
# Epoch 5/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5153 - accuracy: 0.7554 - val_loss: 0.4651 - val_accuracy: 0.7368
# Epoch 6/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5016 - accuracy: 0.7674 - val_loss: 0.4550 - val_accuracy: 0.7653
# Epoch 7/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5042 - accuracy: 0.7593 - val_loss: 0.4532 - val_accuracy: 0.7496
# Epoch 8/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.4989 - accuracy: 0.7633 - val_loss: 0.4536 - val_accuracy: 0.7620
# Epoch 9/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5035 - accuracy: 0.7609 - val_loss: 0.4553 - val_accuracy: 0.7539
# Epoch 10/10
# 492/492 [==============================] - 1s 2ms/step - loss: 0.5023 - accuracy: 0.7588 - val_loss: 0.4567 - val_accuracy: 0.7544
# <keras.callbacks.History at 0x7f879af06f90>



# ####  
# ### B. 다중 분류 DNN 구성
# + 13개 input layer
# + unit 5개 hidden layer
# + dropout
# + unit 4개 hidden layer 
# + dropout
# + 2개 output layser : 이진분류




# 다중분류
# [출처] https://www.educba.com/dnn-neural-network/



# 39개 input layer
# unit 5개 hidden layer
# dropout
# unit 4개 hidden layer 
# dropout
# 2개 output layser : 다중분류

model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(39,)))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))


####  
### 모델 확인
model.summary()
# Model: "sequential_2"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense_6 (Dense)             (None, 5)                 200       
                                                                 
#  dropout_2 (Dropout)         (None, 5)                 0         
                                                                 
#  dense_7 (Dense)             (None, 4)                 24        
                                                                 
#  dropout_3 (Dropout)         (None, 4)                 0         
                                                                 
#  dense_8 (Dense)             (None, 2)                 10        
                                                                 
# =================================================================
# Total params: 234
# Trainable params: 234
# Non-trainable params: 0
# _________________________________________________________________




####  
### 모델 컴파일 – 다중 분류 모델
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 


####  
### 모델 학습
history = model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
          epochs=20, 
          batch_size=16)
# Epoch 1/20
# 308/308 [==============================] - 1s 3ms/step - loss: 0.5507 - accuracy: 0.7322 - val_loss: 0.4708 - val_accuracy: 0.7345
# Epoch 2/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.5011 - accuracy: 0.7351 - val_loss: 0.4540 - val_accuracy: 0.7345
# Epoch 3/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4894 - accuracy: 0.7338 - val_loss: 0.4482 - val_accuracy: 0.7345
# Epoch 4/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4916 - accuracy: 0.7344 - val_loss: 0.4455 - val_accuracy: 0.7345
# Epoch 5/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4850 - accuracy: 0.7340 - val_loss: 0.4420 - val_accuracy: 0.7345
# Epoch 6/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4899 - accuracy: 0.7342 - val_loss: 0.4447 - val_accuracy: 0.7345
# Epoch 7/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4749 - accuracy: 0.7344 - val_loss: 0.4360 - val_accuracy: 0.7345
# Epoch 8/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4779 - accuracy: 0.7342 - val_loss: 0.4374 - val_accuracy: 0.7345
# Epoch 9/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4744 - accuracy: 0.7340 - val_loss: 0.4358 - val_accuracy: 0.7345
# Epoch 10/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4808 - accuracy: 0.7344 - val_loss: 0.4379 - val_accuracy: 0.7345
# Epoch 11/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4761 - accuracy: 0.7344 - val_loss: 0.4379 - val_accuracy: 0.7345
# Epoch 12/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4679 - accuracy: 0.7344 - val_loss: 0.4340 - val_accuracy: 0.7345
# Epoch 13/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4712 - accuracy: 0.7617 - val_loss: 0.4358 - val_accuracy: 0.7824
# Epoch 14/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4790 - accuracy: 0.7489 - val_loss: 0.4388 - val_accuracy: 0.7691
# Epoch 15/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4760 - accuracy: 0.7637 - val_loss: 0.4360 - val_accuracy: 0.7345
# Epoch 16/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4759 - accuracy: 0.7527 - val_loss: 0.4391 - val_accuracy: 0.7710
# Epoch 17/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4739 - accuracy: 0.7562 - val_loss: 0.4375 - val_accuracy: 0.7904
# Epoch 18/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4668 - accuracy: 0.7605 - val_loss: 0.4348 - val_accuracy: 0.7985
# Epoch 19/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4695 - accuracy: 0.7674 - val_loss: 0.4330 - val_accuracy: 0.7899
# Epoch 20/20
# 308/308 [==============================] - 1s 2ms/step - loss: 0.4758 - accuracy: 0.7684 - val_loss: 0.4345 - val_accuracy: 0.7876



####  
### Callback : 조기종료, 모델 저장

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# val_loss 모니터링해서 성능이 5번 지나도록 좋아지지 않으면 조기 종료
early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                           verbose=1, patience=5)

# val_loss 가장 낮은 값을 가질때마다 모델저장
check_point = ModelCheckpoint('best_model.h5', verbose=1, 
                              monitor='val_loss', mode='min', save_best_only=True)


####  
### 모델 학습

history = model.fit(x=X_train, y=y_train, 
          epochs=50 , batch_size=20,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop, check_point])
# Epoch 1/50
# 231/246 [===========================>..] - ETA: 0s - loss: 0.4702 - accuracy: 0.7712
# Epoch 1: val_loss improved from inf to 0.43752, saving model to best_model.h5
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4726 - accuracy: 0.7686 - val_loss: 0.4375 - val_accuracy: 0.7956
# Epoch 2/50
# 241/246 [============================>.] - ETA: 0s - loss: 0.4734 - accuracy: 0.7587
# Epoch 2: val_loss improved from 0.43752 to 0.43419, saving model to best_model.h5
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4727 - accuracy: 0.7603 - val_loss: 0.4342 - val_accuracy: 0.7966
# Epoch 3/50
# 239/246 [============================>.] - ETA: 0s - loss: 0.4754 - accuracy: 0.7692
# Epoch 3: val_loss did not improve from 0.43419
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4733 - accuracy: 0.7698 - val_loss: 0.4343 - val_accuracy: 0.7923
# Epoch 4/50
# 241/246 [============================>.] - ETA: 0s - loss: 0.4699 - accuracy: 0.7645
# Epoch 4: val_loss improved from 0.43419 to 0.43329, saving model to best_model.h5
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4688 - accuracy: 0.7647 - val_loss: 0.4333 - val_accuracy: 0.7975
# Epoch 5/50
# 218/246 [=========================>....] - ETA: 0s - loss: 0.4628 - accuracy: 0.7743
# Epoch 5: val_loss improved from 0.43329 to 0.43201, saving model to best_model.h5
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4633 - accuracy: 0.7694 - val_loss: 0.4320 - val_accuracy: 0.7980
# Epoch 6/50
# 234/246 [===========================>..] - ETA: 0s - loss: 0.4799 - accuracy: 0.7650
# Epoch 6: val_loss did not improve from 0.43201
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4783 - accuracy: 0.7629 - val_loss: 0.4376 - val_accuracy: 0.7980
# Epoch 7/50
# 240/246 [============================>.] - ETA: 0s - loss: 0.4704 - accuracy: 0.7675
# Epoch 7: val_loss did not improve from 0.43201
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4691 - accuracy: 0.7676 - val_loss: 0.4331 - val_accuracy: 0.7961
# Epoch 8/50
# 226/246 [==========================>...] - ETA: 0s - loss: 0.4679 - accuracy: 0.7710
# Epoch 8: val_loss did not improve from 0.43201
# 246/246 [==============================] - 1s 3ms/step - loss: 0.4706 - accuracy: 0.7700 - val_loss: 0.4369 - val_accuracy: 0.7985
# Epoch 9/50
# 222/246 [==========================>...] - ETA: 0s - loss: 0.4734 - accuracy: 0.7716
# Epoch 9: val_loss did not improve from 0.43201
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4734 - accuracy: 0.7674 - val_loss: 0.4357 - val_accuracy: 0.7980
# Epoch 10/50
# 221/246 [=========================>....] - ETA: 0s - loss: 0.4707 - accuracy: 0.7667
# Epoch 10: val_loss did not improve from 0.43201
# 246/246 [==============================] - 1s 2ms/step - loss: 0.4718 - accuracy: 0.7670 - val_loss: 0.4337 - val_accuracy: 0.7999
# Epoch 10: early stopping




####  
## ⑧ 모델 성능 평가

losses = pd.DataFrame(model.history.history)

losses.head()
#   loss	accuracy	val_loss	val_accuracy
# 0	0.472643	0.768605	0.437521	0.795638
# 1	0.472719	0.760268	0.434191	0.796586
# 2	0.473267	0.769825	0.434259	0.792319
# 3	0.468788	0.764742	0.433287	0.797534
# 4	0.463253	0.769418	0.432012	0.798009




### 성능 시각화
losses[['loss','val_loss']].plot()


losses[['loss','val_loss', 'accuracy','val_accuracy']].plot()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc'])
plt.show()



### 성능 평가
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


pred = model.predict(X_test)

pred.shape
# (2109, 2)


y_pred = np.argmax(pred, axis=1)


# 정확도 80%
accuracy_score(y_test, y_pred)
# 0.7999051683262209


# 재현율 성능이 좋지 않다
recall_score(y_test, y_pred)
# 0.49107142857142855



##### <font color=blue>**[문제] accuracy, recall, precision 성능 한번에 출력해보세요 </font>
# accuracy, recall, precision 성능 한번에 보기
#               precision    recall  f1-score   support

#            0       0.83      0.91      0.87      1549
#            1       0.67      0.49      0.57       560

#     accuracy                           0.80      2109
#    macro avg       0.75      0.70      0.72      2109
# weighted avg       0.79      0.80      0.79      2109

