import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="IVA - Image Visualization",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        background: #202124;
        color: white;
        padding: 25px;
        text-align: center;
        border-radius: 10px;
    }

    .main-title h1 {
        margin: 0;
        font-size: 30px;
    }

    .main-title p {
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>Image Visualization & Analysis</h1>
    <p>Spatial Domain Method - Gradient Operators</p>
</div>
""", unsafe_allow_html=True)

st.write("")

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

operator = st.selectbox(
    "Select Gradient Operator",
    [
        "Sobel Operator",
        "Prewitt Operator",
        "Roberts Operator",
        "Laplacian Operator"
    ]
)


def apply_gradient(image, operator):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if operator == "Sobel Operator":

        sobel_x = cv2.Sobel(
            gray, cv2.CV_64F, 1, 0, ksize=3
        )

        sobel_y = cv2.Sobel(
            gray, cv2.CV_64F, 0, 1, ksize=3
        )

        magnitude = cv2.magnitude(
            sobel_x.astype(np.float32),
            sobel_y.astype(np.float32)
        )

        return cv2.convertScaleAbs(magnitude)

    elif operator == "Prewitt Operator":

        kernel_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32)

        kernel_y = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float32)

        gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

        magnitude = cv2.magnitude(gx, gy)

        return cv2.convertScaleAbs(magnitude)

    elif operator == "Roberts Operator":

        kernel_x = np.array([
            [1, 0],
            [0, -1]
        ], dtype=np.float32)

        kernel_y = np.array([
            [0, 1],
            [-1, 0]
        ], dtype=np.float32)

        gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

        magnitude = cv2.magnitude(gx, gy)

        return cv2.convertScaleAbs(magnitude)

    elif operator == "Laplacian Operator":

        result = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        return cv2.convertScaleAbs(result)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is not None:

        processed = apply_gradient(
            image,
            operator
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")

            original_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                original_rgb,
                use_container_width=True
            )

        with col2:
            st.subheader(operator)

            st.image(
                processed,
                use_container_width=True
            )

        st.divider()

        st.subheader("🔍 Gradient Operator Information")

        if operator == "Sobel Operator":

            st.write(
                "**Sobel Operator** detects edges by calculating "
                "horizontal and vertical intensity changes."
            )

            st.write(
                "It uses two 3 × 3 kernels for detecting "
                "X and Y directions."
            )

        elif operator == "Prewitt Operator":

            st.write(
                "**Prewitt Operator** is used for detecting "
                "edges and boundaries in an image."
            )

            st.write(
                "It uses horizontal and vertical 3 × 3 kernels."
            )

        elif operator == "Roberts Operator":

            st.write(
                "**Roberts Operator** is a simple gradient "
                "operator that detects edges using 2 × 2 kernels."
            )

        elif operator == "Laplacian Operator":

            st.write(
                "**Laplacian Operator** is a second-order "
                "derivative operator that detects regions "
                "of rapid intensity change."
            )
