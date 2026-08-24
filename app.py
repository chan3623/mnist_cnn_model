# app.py
import os

import numpy as np
import streamlit as st
import torch
import torchvision.transforms.v2 as transform
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torch import nn


# ------------------------------------------------------------
# 1. 모델 클래스 정의 (학습 시 정의된 것과 완전히 동일해야 함)
# ------------------------------------------------------------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# 2. 모델 로드 함수 (Streamlit 캐싱 적용)
# ------------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN()
    # weights_only=True 옵션을 주어 보안 경고 방지 (학습 코드와 동일)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()  # 평가 모드 전환
    return model, device


# ------------------------------------------------------------
# 3. 이미지 전처리 함수
# ------------------------------------------------------------
def preprocess_image(image):
    # PIL 이미지를 입력받아 학습 모델과 동일한 변환 적용
    mnist_transform = transform.Compose(
        [
            transform.ToImage(),
            transform.ToDtype(torch.float32, scale=True),
        ]
    )
    tensor_img = mnist_transform(image)
    # [1, 28, 28] 형태의 텐서에 Batch 차원 추가 -> [1, 1, 28, 28]
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img


# ------------------------------------------------------------
# 4. Streamlit UI 구성
# ------------------------------------------------------------
st.title("🖌️ MNIST CNN 숫자 인식기")
st.write("학습된 PyTorch 모델을 불러와 손글씨 숫자를 인식해보세요.")

# 저장된 모델 폴더 확인
model_dir = "saved_models"
if not os.path.exists(model_dir):
    st.error(f"'{model_dir}' 폴더가 없습니다. 모델을 먼저 학습하여 저장해주세요.")
    st.stop()

# .pth 확장자를 가진 모델 파일 목록 불러오기
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
if not model_files:
    st.error(f"'{model_dir}' 폴더 내에 모델 파일(.pth)이 없습니다.")
    st.stop()

# 사이드바에서 사용할 모델 선택
st.sidebar.header("설정")
selected_model = st.sidebar.selectbox(
    "사용할 모델을 선택하세요:", sorted(model_files, reverse=True)
)
model_path = os.path.join(model_dir, selected_model)

# 선택된 모델 로드
try:
    model, device = load_model(model_path)
    st.sidebar.success(f"모델 로드 완료! (사용 장치: {device})")
except Exception as e:
    st.sidebar.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 입력 방식 선택 (캔버스 vs 업로드)
input_method = st.radio(
    "테스트 방식을 선택하세요:", ("캔버스에 직접 그리기", "이미지 파일 업로드")
)

processed_img = None

if input_method == "캔버스에 직접 그리기":
    st.write("아래 검은 바탕에 마우스로 **0~9 사이의 숫자**를 그려주세요.")

    # 그리기 캔버스 위젯
    canvas_result = st_canvas(
        fill_color="#000000",  # 채우기 색상
        stroke_width=18,  # 브러시 두께 (28x28 축소 시 잘 보이도록 두껍게 설정)
        stroke_color="#FFFFFF",  # 브러시 색상 (MNIST는 배경 검정, 글씨 흰색)
        background_color="#000000",  # 배경색
        width=280,  # 캔버스 너비
        height=280,  # 캔버스 높이
        drawing_mode="freedraw",
        key="canvas",
    )

    # 캔버스에 그림이 그려진 경우 이미지 추출
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data.astype(np.uint8)
        # RGBA 이미지를 Grayscale(L)로 변환
        pil_img = Image.fromarray(img_array).convert("L")
        # MNIST 모델에 맞게 28x28 사이즈로 축소 (LANCZOS 필터 사용)
        pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)

        # 완전히 비어있는(검은색) 이미지가 아닐 때만 처리 변수에 저장
        if np.max(np.array(pil_img)) > 0:
            processed_img = pil_img
            st.image(
                pil_img.resize((140, 140)),
                caption="모델 입력용 크기(28x28)",
                use_container_width=False,
            )

else:
    uploaded_file = st.file_uploader(
        "숫자 이미지 업로드 (어두운 배경에 밝은 숫자 권장)", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        # 업로드된 이미지를 흑백으로 열고 28x28로 변환
        pil_img = Image.open(uploaded_file).convert("L")
        pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        processed_img = pil_img
        st.image(
            pil_img.resize((140, 140)),
            caption="모델 입력용 크기(28x28)",
            use_container_width=False,
        )

# ------------------------------------------------------------
# 5. 예측 및 결과 시각화
# ------------------------------------------------------------
if st.button("결과 예측하기", type="primary"):
    if processed_img is None:
        st.warning("먼저 숫자를 그리거나 이미지를 업로드해주세요.")
    else:
        # 이미지를 Tensor로 변환하고 디바이스로 이동
        input_tensor = preprocess_image(processed_img).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            # Logit 출력을 Softmax 함수에 통과시켜 확률로 변환
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item() * 100

        # 결과 출력
        st.subheader(f"💡 모델의 예측 결과: **{predicted_class}**")
        st.write(f"정답일 확률: **{confidence:.2f}%**")

        # 0~9까지 클래스별 확률 바 차트 (막대 그래프) 생성
        prob_array = probabilities.cpu().numpy()
        chart_data = {str(i): prob_array[i] for i in range(10)}
        st.write("클래스별 확률 분포:")
        st.bar_chart(chart_data)
