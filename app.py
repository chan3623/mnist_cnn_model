# app.py
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms.v2 as transform
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torch import nn

# ------------------------------------------------------------
# 0. 페이지 기본 설정 및 커스텀 CSS (UI/UX 개선)
# ------------------------------------------------------------
st.set_page_config(
    page_title="MNIST 숫자 인식기",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* 기본 버튼 스타일링 (크고 둥글게) */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 0;
            transition: all 0.3s;
        }
        /* 예측 결과 텍스트 강조용 컨테이너 */
        .result-box {
            background-color: #f1f8ff;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .result-number {
            font-size: 100px;
            font-weight: 900;
            color: #007bff;
            line-height: 1;
            margin: 10px 0;
        }
        .result-label {
            font-size: 20px;
            color: #495057;
            font-weight: 600;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 1. 모델 클래스 정의 (학습 시 정의된 것과 완전히 동일)
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
# 2. 모델 로드 함수 (캐싱)
# ------------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN()
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, device


# ------------------------------------------------------------
# 3. 이미지 전처리 함수
# ------------------------------------------------------------
def preprocess_image(image):
    mnist_transform = transform.Compose(
        [
            transform.ToImage(),
            transform.ToDtype(torch.float32, scale=True),
        ]
    )
    tensor_img = mnist_transform(image)
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img


# ------------------------------------------------------------
# 4. Streamlit UI 구성 - 메인 헤더
# ------------------------------------------------------------
st.title("✨ AI 손글씨 숫자 인식기")
st.markdown("학습된 PyTorch 모델을 사용하여 0부터 9까지의 숫자를 판별합니다.")
st.divider()

# 모델 폴더 확인
model_dir = "saved_models"
if not os.path.exists(model_dir):
    st.error(f"'{model_dir}' 폴더가 없습니다. 모델을 먼저 학습하여 저장해주세요.")
    st.stop()

model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
if not model_files:
    st.error(f"'{model_dir}' 폴더 내에 모델 파일(.pth)이 없습니다.")
    st.stop()

# 사이드바 설정 영역
with st.sidebar:
    st.header("⚙️ 환경 설정")
    selected_model = st.selectbox("사용할 모델 선택", sorted(model_files, reverse=True))
    model_path = os.path.join(model_dir, selected_model)

    try:
        model, device = load_model(model_path)
        st.success(f"✅ 모델 로드 성공!\n\n(장치: {device})")
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()

    st.markdown("---")
    st.markdown(
        "**Tip:** 캔버스에 숫자를 화면에 꽉 차게, 중앙에 그릴수록 인식률이 올라갑니다."
    )

# ------------------------------------------------------------
# 5. 입력 영역 (카드 레이아웃 적용)
# ------------------------------------------------------------
input_container = st.container(border=True)
with input_container:
    st.subheader("📝 입력 방식 선택")
    input_method = st.radio(
        "테스트 방식을 선택하세요:",
        ("✏️ 캔버스에 직접 그리기", "📁 이미지 파일 업로드"),
        horizontal=True,  # 라디오 버튼 가로 배치
        label_visibility="collapsed",
    )

    processed_img = None

    if input_method == "✏️ 캔버스에 직접 그리기":
        col1, col2 = st.columns([1.5, 1], gap="large")

        with col1:
            st.markdown("##### 0~9 사이의 숫자를 그려주세요")
            canvas_result = st_canvas(
                fill_color="#000000",
                stroke_width=20,  # 브러시를 살짝 더 두껍게 (인식률 향상)
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas",
            )

        with col2:
            st.markdown("##### AI 입력 미리보기")
            if canvas_result.image_data is not None:
                img_array = canvas_result.image_data.astype(np.uint8)
                pil_img = Image.fromarray(img_array).convert("L")
                pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)

                if np.max(np.array(pil_img)) > 0:
                    processed_img = pil_img
                    # 미리보기 이미지를 보기 좋게 테두리와 함께 출력
                    st.image(pil_img.resize((150, 150)), caption="28x28 해상도 변환됨")
                else:
                    st.info("숫자를 그리면 미리보기가 생성됩니다.")

    else:
        uploaded_file = st.file_uploader(
            "어두운 배경에 밝은 숫자가 있는 이미지를 업로드하세요",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="원본 이미지", width=150)
            with col2:
                pil_img = Image.open(uploaded_file).convert("L")
                pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
                processed_img = pil_img
                st.image(pil_img.resize((150, 150)), caption="모델 입력용 변환 (28x28)")

# ------------------------------------------------------------
# 6. 예측 실행 및 결과 출력 영역
# ------------------------------------------------------------
st.write("")  # 여백
if st.button("🚀 AI 결과 예측하기", type="primary"):
    if processed_img is None:
        st.warning("⚠️ 먼저 숫자를 그리거나 이미지를 업로드해주세요.")
    else:
        # 진행 상태 스피너 표시
        with st.spinner("AI가 이미지를 분석하고 있습니다..."):
            input_tensor = preprocess_image(processed_img).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100
                prob_array = probabilities.cpu().numpy()

        # 결과 컨테이너 (시각적 강조)
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">모델이 예측한 숫자는</div>
                <div class="result-number">{predicted_class}</div>
                <div class="result-label">확률: {confidence:.2f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.balloons()  # 정답 도출 시 풍선 이펙트 추가

        # 확률 분포 차트 영역
        chart_container = st.container(border=True)
        with chart_container:
            st.subheader("📊 클래스별 세부 확률 분포")

            # Pandas DataFrame으로 변환하여 Streamlit bar_chart를 예쁘게 출력
            df_probs = pd.DataFrame(
                {"클래스 (숫자)": [str(i) for i in range(10)], "확률": prob_array}
            ).set_index("클래스 (숫자)")

            st.bar_chart(df_probs, color="#007bff", height=250)
