"""
딥러닝 초보자를 위한 CNN 실습 - 4단계: CPU vs GPU 학습 속도 비교
동일한 모델, 동일한 데이터로 CPU와 GPU에서 각각 몇 epoch을 학습해서
걸린 시간을 비교합니다.

참고: 로컬 PC에 GPU가 없다면 Google Colab(무료 GPU 제공)에서 실행해보세요.
      Colab에서는 [런타임] -> [런타임 유형 변경] -> [GPU] 로 설정하면 됩니다.

실행:
    python 04_benchmark_cpu_gpu.py
"""

import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

BENCH_EPOCHS = 3
BATCH_SIZE = 64


def build_model():
    """01_train.py와 동일한 구조의 새 모델을 매번 새로 만듭니다.
    (같은 모델 객체를 재사용하면 이미 학습된 가중치가 남아있어 비교가 불공평해집니다)"""
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_on_device(device_name, x_train, y_train, x_test, y_test):
    print(f"\n=== {device_name} 에서 학습 시작 ===")
    with tf.device(device_name):
        model = build_model()
        start = time.time()
        model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=BENCH_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )
        elapsed = time.time() - start
    print(f"{device_name} 학습 시간: {elapsed:.2f}초 ({BENCH_EPOCHS} epoch 기준)")
    return elapsed


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 벤치마크는 시간이 오래 걸릴 수 있으므로 학습 데이터 일부(10,000장)만 사용합니다.
    x_train_small = x_train[:10000]
    y_train_small = y_train[:10000]

    gpus = tf.config.list_physical_devices('GPU')

    results = {}

    # CPU 벤치마크 (항상 실행)
    results['CPU'] = run_on_device('/CPU:0', x_train_small, y_train_small, x_test, y_test)

    # GPU 벤치마크 (GPU가 있을 때만 실행)
    if gpus:
        results['GPU'] = run_on_device('/GPU:0', x_train_small, y_train_small, x_test, y_test)
    else:
        print("\nGPU가 감지되지 않아 GPU 벤치마크는 건너뜁니다.")
        print("GPU 환경(Google Colab 등)에서 다시 실행하면 CPU와 GPU 비교 결과를 볼 수 있습니다.")

    # 결과 출력
    print("\n=== 결과 요약 ===")
    for device, elapsed in results.items():
        print(f"{device}: {elapsed:.2f}초")

    if 'GPU' in results:
        speedup = results['CPU'] / results['GPU']
        print(f"GPU가 CPU보다 약 {speedup:.1f}배 빠릅니다.")

    # 막대 그래프로 저장
    plt.figure(figsize=(5, 4))
    plt.bar(results.keys(), results.values(), color=['gray', 'orange'][:len(results)])
    plt.ylabel('학습 시간 (초)')
    plt.title(f'CPU vs GPU 학습 속도 비교\n({BENCH_EPOCHS} epoch, 10,000장 기준)')
    for i, (device, elapsed) in enumerate(results.items()):
        plt.text(i, elapsed, f"{elapsed:.1f}s", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("outputs/cpu_gpu_benchmark.png", dpi=120)
    print("그래프 저장 완료: outputs/cpu_gpu_benchmark.png")


if __name__ == "__main__":
    main()
