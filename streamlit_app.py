import cv2
from extractor import SpectralExtractor
import numpy as np
from skimage import data
from sklearn.svm import SVC
import streamlit as st

st.set_page_config(page_title="DeepSpectra Scanner", page_icon="🔬")

st.title("🔬 DeepSpectra Forensic Scanner (v2.0)")
st.write(
    "Upload any facial image or screenshot to analyze frequency-domain"
    " perturbations."
)


# 1. إعداد المحرك والتدريب السريع في الذاكرة
@st.cache_resource
def load_engine():
    ext = SpectralExtractor(target_radius=120)
    base_img = data.astronaut()
    h, w, _ = base_img.shape
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    X, y = [], []
    for b in [0.8, 1.0, 1.2]:
        real_s = np.clip(base_gray.astype(float) * b, 0, 255).astype(np.uint8)
        X.append(ext.process_image(real_s))
        y.append(0)

    for amp in [5, 8, 12]:
        noise = (
            amp
            * np.sin(2 * np.pi * grid_x / 8)
            * np.sin(2 * np.pi * grid_y / 8)
        )
        fake_s = np.clip(base_gray.astype(float) + noise, 0, 255).astype(
            np.uint8
        )
        X.append(ext.process_image(fake_s))
        y.append(1)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)
    return ext, clf


extractor, model = load_engine()

# 2. واجهة رفع الصور
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # حساب الترددات FFT
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    norm_spectrum = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # الفحص
    feats = extractor.process_image(image).reshape(1, -1)
    pred = model.predict(feats)[0]
    prob = model.predict_proba(feats)[0]

    # عرض النتائج
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            use_column_width=True,
        )
    with col2:
        st.image(
            heatmap_rgb,
            caption="FFT Frequency Heatmap",
            use_column_width=True,
        )

    if pred == 0:
        st.success(
            f"✅ Verdict: REAL IMAGE (Confidence: {prob[0]*100:.1f}%)"
        )
    else:
        st.error(
            f"🚨 Verdict: DEEPFAKE DETECTED (Confidence: {prob[1]*100:.1f}%)"
        )
