"""
딥러닝 초보자를 위한 CNN 실습 - 2단계: 레이어별 액티베이션 맵(특징 맵) 시각화
학습된 모델이 이미지를 볼 때, 각 Conv 레이어가 어떤 특징을 뽑아내는지
직접 눈으로 확인합니다.

실행 전 준비:
    01_train.py를 먼저 실행해서 outputs/model/cnn_cifar10.keras 가 있어야 합니다.

실행:
    python 02_visualize_activations.py
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs("outputs/activations", exist_ok=True)

# ------------------------------------------------------------------
# 1. 학습된 모델과 테스트 데이터 로드
# ------------------------------------------------------------------
model = load_model("outputs/model/cnn_cifar10.keras")
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

# 시각화할 샘플 이미지 한 장 선택 (인덱스를 바꾸면 다른 이미지로 확인 가능)
sample_idx = 0
sample_img = x_test[sample_idx:sample_idx + 1]  # shape: (1, 32, 32, 3)
true_label = CLASS_NAMES[int(y_test[sample_idx])]

# ------------------------------------------------------------------
# 2. Conv 레이어들의 출력을 모두 뽑아내는 서브 모델 생성
#    -> 원래 모델의 입력은 그대로 두고, 출력만 중간 레이어들로 바꿔치기
# ------------------------------------------------------------------
conv_layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
print(f"액티베이션을 뽑을 레이어: {conv_layer_names}")

layer_outputs = [model.get_layer(name).output for name in conv_layer_names]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(sample_img)
if len(conv_layer_names) == 1:
    activations = [activations]

# ------------------------------------------------------------------
# 3. 원본 입력 이미지 저장
# ------------------------------------------------------------------
plt.figure(figsize=(2, 2))
plt.imshow(sample_img[0])
plt.title(f"input: {true_label}")
plt.axis('off')
plt.savefig("outputs/activations/00_input.png", dpi=120, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------
# 4. 레이어별 액티베이션 맵을 그리드 이미지로 저장
#    각 레이어에서 대표로 채널 8개만 뽑아서 보여줍니다.
# ------------------------------------------------------------------
NUM_CHANNELS_TO_SHOW = 8

for layer_name, activation in zip(conv_layer_names, activations):
    num_channels = activation.shape[-1]
    show_n = min(NUM_CHANNELS_TO_SHOW, num_channels)

    fig, axes = plt.subplots(1, show_n, figsize=(show_n * 1.5, 1.8))
    if show_n == 1:
        axes = [axes]

    for i in range(show_n):
        feature_map = activation[0, :, :, i]
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f"ch{i}", fontsize=8)

    fig.suptitle(
        f"{layer_name}  (특징 맵 크기: {activation.shape[1]}x{activation.shape[2]}, "
        f"총 채널 수: {num_channels})",
        fontsize=10,
    )
    plt.tight_layout()
    save_path = f"outputs/activations/{layer_name}.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"저장 완료: {save_path}")

print("\n모든 레이어의 액티베이션 맵 저장이 끝났습니다. outputs/activations 폴더를 확인하세요.")
print("레이어가 깊어질수록(conv1 -> conv2 -> conv3) 특징 맵의 가로/세로 크기는 작아지고,")
print("표현하는 특징은 더 추상적으로(선/경계 -> 무늬/질감 -> 사물의 부분) 바뀌는 것을 관찰할 수 있습니다.")
