"""
딥러닝 초보자를 위한 CNN 실습 - 3단계: 테스트 결과 시각화
학습된 모델이 실제로 이미지를 얼마나 잘 "검출(분류)"하는지 눈으로 확인합니다.
정답이면 초록색 제목, 오답이면 빨간색 제목으로 표시됩니다.

실행 전 준비:
    01_train.py를 먼저 실행해서 outputs/model/cnn_cifar10.keras 가 있어야 합니다.

실행:
    python 03_test_predictions.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs("outputs", exist_ok=True)

model = load_model("outputs/model/cnn_cifar10.keras")
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test_norm = x_test.astype("float32") / 255.0
y_test = y_test.flatten()

# ------------------------------------------------------------------
# 1. 테스트셋에서 무작위로 16장을 뽑아 예측 수행
# ------------------------------------------------------------------
np.random.seed(42)
sample_indices = np.random.choice(len(x_test_norm), size=16, replace=False)

sample_images = x_test_norm[sample_indices]
sample_labels = y_test[sample_indices]

predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)

# ------------------------------------------------------------------
# 2. 4x4 그리드로 결과 시각화
# ------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

correct_count = 0
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i])
    ax.axis('off')

    true_name = CLASS_NAMES[sample_labels[i]]
    pred_name = CLASS_NAMES[predicted_labels[i]]
    conf = confidences[i] * 100

    is_correct = sample_labels[i] == predicted_labels[i]
    correct_count += int(is_correct)
    color = 'green' if is_correct else 'red'

    ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.0f}%)",
                 fontsize=9, color=color)

fig.suptitle(f"{correct_count} / 16 correct", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/test_predictions.png", dpi=120, bbox_inches='tight')
print("저장 완료: outputs/test_predictions.png")

# ------------------------------------------------------------------
# 3. 전체 테스트셋 정확도 출력
# ------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test_norm, y_test, verbose=0)
print(f"\n전체 테스트셋(10,000장) 정확도: {test_acc * 100:.2f}%")
