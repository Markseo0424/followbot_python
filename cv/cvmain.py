import numpy as np
import cv2
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt

onnx_model_path = '../onnx/Followbot_pi.onnx'  # 모델 경로를 지정
session = ort.InferenceSession(onnx_model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 웹캠을 초기화합니다 (웹캠 인덱스 0번을 사용)
cap = cv2.VideoCapture(0)


def increase_saturation(frame, saturation_scale=1):
    # BGR 이미지를 HSV 색 공간으로 변환합니다
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV에서 채도(Saturation) 값을 가져옵니다 (2번째 채널)
    h, s, v = cv2.split(hsv)

    # 채도 값에 스케일을 곱하여 증가시킵니다
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)

    # 다시 HSV 이미지를 합칩니다
    hsv_adjusted = cv2.merge([h, s, v])

    # HSV 이미지를 다시 BGR로 변환합니다
    frame_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    return frame_adjusted


def process(frame) :
    # frame = torch.tensor(frame)
    # frame = torch.permute(frame, (2,0,1))
    # x = frame.to(device)
    frame = np.array(frame).transpose(2, 0, 1).astype(np.float32)
    frame = frame.reshape(1, 3, 224, 224)
    frame = frame[:, ::-1]
    results = session.run([output_name], {input_name: frame})

    r, l = results[0][0][:2]
    print(r, l)

    return r, l



# 프레임이 올바르게 열렸는지 확인합니다
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 웹캠에서 프레임을 읽어옵니다
    ret, frame = cap.read()

    # 프레임을 정상적으로 읽었는지 확인
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 원본 프레임의 높이와 너비를 가져옵니다
    height, width, _ = frame.shape

    # 정사각형으로 만들기 위해 중앙 부분을 크롭합니다
    min_dim = min(height, width)

    # 중앙을 기준으로 크롭
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    square_frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # 크롭한 이미지를 224x224로 리사이즈합니다
    resized_frame = cv2.resize(square_frame, (224, 224))

    resized_frame = increase_saturation(resized_frame)

    # mean, std = resized_frame.mean(), resized_frame.std()
    #
    # normalized = (resized_frame - mean)/std
    # normalized_frame = normalized * 0.2 + 0.5

    normalized_frame = resized_frame / 255

    r, l = process(normalized_frame)
    # plt.scatter(r, l)
    # plt.pause(0.01)

    # normalized_frame += np.random.randn(*normalized_frame.shape) * 0.1

    # 리사이즈된 프레임을 화면에 출력합니다
    cv2.imshow('Square Resized Frame (224x224)', normalized_frame)

    # 'q' 키를 누르면 루프를 종료합니다
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스를 해제하고 창을 닫습니다
cap.release()
cv2.destroyAllWindows()
