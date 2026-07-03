"""
딥러닝 초보자를 위한 CNN 실습 - 1단계: 모델 학습
CIFAR-10 데이터셋(32x32 컬러 이미지, 10개 클래스)을 사용해
이미지가 어떤 사물인지 "검출(분류)"하는 CNN 모델을 학습합니다.

실행:
    python 01_train.py
"""

import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/model", exist_ok=True)

# ------------------------------------------------------------------
# 1. GPU 사용 가능 여부 확인
# ------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
print(f"사용 가능한 GPU 개수: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("  GPU를 찾지 못했습니다. CPU로 학습을 진행합니다.")

# ------------------------------------------------------------------
# 2. 데이터셋 로드
#    CIFAR-10: 10개 클래스, 32x32 컬러 이미지, 전체 약 170MB
#    tf.keras 내장 함수로 처음 실행 시 자동 다운로드됩니다.
# ------------------------------------------------------------------
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 픽셀 값을 0~1 사이로 정규화 (학습이 훨씬 안정적으로 됩니다)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"학습 데이터: {x_train.shape}, 테스트 데이터: {x_test.shape}")

# ------------------------------------------------------------------
# 3. 간단한 CNN 모델 구성
#    Conv2D + MaxPooling을 3번 반복 -> Flatten -> Dense -> 10개 클래스 분류
#    각 Conv2D 레이어에 이름을 붙여서, 2단계 스크립트에서 액티베이션 맵을
#    쉽게 뽑아낼 수 있도록 합니다.
# ------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),

    layers.Flatten(),
    layers.Dense(64, activation='relu', name='dense1'),
    layers.Dense(10, activation='softmax', name='predictions'),  # 10개 클래스 중 하나를 "검출"
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# ------------------------------------------------------------------
# 4. 학습 (걸린 시간을 측정해서 4단계 CPU/GPU 비교와 비교해볼 수 있습니다)
# ------------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 64

start = time.time()
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)
elapsed = time.time() - start
print(f"\n총 학습 시간: {elapsed:.2f}초 ({EPOCHS} epoch 기준)")

# ------------------------------------------------------------------
# 5. 모델 저장 (2, 3단계 스크립트에서 재사용)
# ------------------------------------------------------------------
model.save("outputs/model/cnn_cifar10.keras")
print("모델 저장 완료: outputs/model/cnn_cifar10.keras")

# ------------------------------------------------------------------
# 6. 학습 곡선(정확도/손실) 그래프 저장
# ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history.history['accuracy'], label='train')
axes[0].plot(history.history['val_accuracy'], label='val')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'], label='train')
axes[1].plot(history.history['val_loss'], label='val')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/training_curve.png", dpi=120)
print("학습 곡선 저장 완료: outputs/training_curve.png")
