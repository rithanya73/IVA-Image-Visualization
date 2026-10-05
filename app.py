import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Interactive Vision Analysis",
    page_icon="👁️",
    layout="wide"
)


# =========================================================
# GLOBAL PURPLE + WHITE MARBLE BACKGROUND
# =========================================================

st.markdown(
    """
    <style>

    /* =========================================
       MAIN PURPLE MARBLE BACKGROUND
       ========================================= */

    .stApp {
        background:
            radial-gradient(
                ellipse 55% 40% at 15% 15%,
                rgba(255, 255, 255, 0.30),
                transparent 60%
            ),
            radial-gradient(
                ellipse 45% 35% at 85% 80%,
                rgba(255, 255, 255, 0.22),
                transparent 60%
            ),
            linear-gradient(
                135deg,
                #12001f 0%,
                #260044 18%,
                #4b0878 38%,
                #25003f 55%,
                #641394 73%,
                #30004f 88%,
                #10001c 100%
            );

        background-attachment: fixed;
        min-height: 100vh;
    }


    /* =========================================
       WHITE MARBLE VEINS
       ========================================= */

    .stApp::before {
        content: "";
        position: fixed;
        inset: -50%;

        background:
            repeating-linear-gradient(
                115deg,
                transparent 0px,
                transparent 95px,
                rgba(255,255,255,0.08) 100px,
                rgba(255,255,255,0.28) 106px,
                rgba(255,255,255,0.08) 113px,
                transparent 120px,
                transparent 220px
            );

        transform: rotate(-4deg);
        opacity: 0.75;
        pointer-events: none;
        z-index: 0;
    }


    /* =========================================
       SOFT MARBLE PATCHES
       ========================================= */

    .stApp::after {
        content: "";
        position: fixed;
        inset: -30%;

        background:
            radial-gradient(
                ellipse at 20% 40%,
                transparent 0%,
                transparent 25%,
                rgba(255,255,255,0.12) 27%,
                transparent 34%
            ),
            radial-gradient(
                ellipse at 75% 25%,
                transparent 0%,
                transparent 22%,
                rgba(255,255,255,0.10) 24%,
                transparent 31%
            ),
            radial-gradient(
                ellipse at 65% 75%,
                transparent 0%,
                transparent 24%,
                rgba(255,255,255,0.14) 26%,
                transparent 33%
            );

        transform: rotate(12deg);
        pointer-events: none;
        z-index: 0;
    }


    /* =========================================
       KEEP STREAMLIT CONTENT ABOVE BACKGROUND
       ========================================= */

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: transparent;
    }

    .main,
    .block-container {
        position: relative;
        z-index: 1;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "tm_result" not in st.session_state:
    st.session_state.tm_result = None

if "tm_data" not in st.session_state:
    st.session_state.tm_data = None

if "tm_scan_done" not in st.session_state:
    st.session_state.tm_scan_done = False

if "vj_result" not in st.session_state:
    st.session_state.vj_result = None

if "df_result" not in st.session_state:
    st.session_state.df_result = None

if "fn_result" not in st.session_state:
    st.session_state.fn_result = None


# =========================================================
# HOME PAGE
# =========================================================

def show_home():

    st.title("👁️ Interactive Vision Analysis")

    st.subheader("Learn. Visualize. Try.")

    st.write(
        "Explore computer vision and facial analysis techniques "
        "through interactive visual demonstrations."
    )

    st.write("")

    st.markdown("### Learning Flow")

    flow = st.columns(9)

    with flow[0]:
        st.info("📖\n\nLearn")

    with flow[1]:
        st.write("→")

    with flow[2]:
        st.info("👁️\n\nVisualize")

    with flow[3]:
        st.write("→")

    with flow[4]:
        st.info("🧪\n\nTry")

    with flow[5]:
        st.write("→")

    with flow[6]:
        st.info("🔍\n\nCompare")

    with flow[7]:
        st.write("→")

    with flow[8]:
        st.info("🧠\n\nQuiz")

    st.write("")
    st.divider()

    st.markdown("### Computer Vision Techniques")

    col1, col2 = st.columns(2, gap="medium")

    with col1:

        st.subheader("🎯 Template Matching")

        st.write(
            "Find a small template image inside a larger image "
            "using OpenCV template matching."
        )

        if st.button(
            "Explore Template Matching",
            key="home_tm",
            use_container_width=True
        ):

            st.session_state.page = "template_matching"
            st.session_state.tm_result = None
            st.session_state.tm_data = None
            st.session_state.tm_scan_done = False

            st.rerun()

    with col2:

        st.subheader("🧑 Viola–Jones")

        st.write(
            "Detect human faces using Haar-like features, "
            "integral images, AdaBoost and cascade classifiers."
        )

        if st.button(
            "Explore Viola–Jones",
            key="home_vj",
            use_container_width=True
        ):

            st.session_state.page = "viola_jones"
            st.session_state.vj_result = None

            st.rerun()

    st.write("")

    col3, col4 = st.columns(2, gap="medium")

    with col3:

        st.subheader("🤖 DeepFace")

        st.write(
            "Compare two face images and determine whether "
            "they belong to the same person."
        )

        if st.button(
            "Explore DeepFace",
            key="home_deepface",
            use_container_width=True
        ):

            st.session_state.page = "deepface"
            st.session_state.df_result = None

            st.rerun()

    with col4:

        st.subheader("🔗 FaceNet")

        st.write(
            "Convert a face into a numerical embedding "
            "using the FaceNet deep-learning model."
        )

        if st.button(
            "Explore FaceNet",
            key="home_facenet",
            use_container_width=True
        ):

            st.session_state.page = "facenet"
            st.session_state.fn_result = None

            st.rerun()


# =========================================================
# TEMPLATE MATCHING
# =========================================================

def show_template_matching():

    if st.button(
        "← Back to Techniques",
        key="tm_back"
    ):

        st.session_state.page = "home"
        st.session_state.tm_result = None
        st.session_state.tm_data = None
        st.session_state.tm_scan_done = False

        st.rerun()

    st.title("🎯 Template Matching")

    st.write(
        "Find a smaller template image inside a larger image "
        "using OpenCV template matching."
    )

    st.divider()

    left, right = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with left:

        st.subheader("📷 Image Input")

        main_file = st.file_uploader(
            "Upload Main Image",
            type=["jpg", "jpeg", "png"],
            key="tm_main_upload"
        )

        template_file = st.file_uploader(
            "Upload Template Image",
            type=["jpg", "jpeg", "png"],
            key="tm_template_upload"
        )

        if main_file is not None and template_file is not None:

            main_image = Image.open(
                main_file
            ).convert("RGB")

            template_image = Image.open(
                template_file
            ).convert("RGB")

            main_array = np.array(main_image)
            template_array = np.array(template_image)

            main_height, main_width = main_array.shape[:2]

            template_height, template_width = (
                template_array.shape[:2]
            )

            if (
                template_height > main_height
                or template_width > main_width
            ):

                st.error(
                    "❌ Template image must be smaller than "
                    "the main image."
                )

            else:

                image_col1, image_col2 = st.columns(2)

                with image_col1:

                    st.image(
                        main_array,
                        caption="Main Image",
                        width=300
                    )

                with image_col2:

                    st.image(
                        template_array,
                        caption="Template",
                        width=220
                    )

                st.write("")

                find_match = st.button(
                    "🔍 Find Match",
                    key="find_match",
                    use_container_width=True
                )

                if find_match:

                    main_gray = cv2.cvtColor(
                        main_array,
                        cv2.COLOR_RGB2GRAY
                    )

                    template_gray = cv2.cvtColor(
                        template_array,
                        cv2.COLOR_RGB2GRAY
                    )

                    result = cv2.matchTemplate(
                        main_gray,
                        template_gray,
                        cv2.TM_CCOEFF_NORMED
                    )

                    result_h, result_w = result.shape

                    st.divider()

                    st.subheader("🔎 Automatic Scan")

                    st.caption(
                        "OpenCV evaluates every valid position "
                        "row by row."
                    )

                    scan_placeholder = st.empty()

                    total_positions = (
                        result_h * result_w
                    )

                    max_frames = 80

                    step = max(
                        1,
                        total_positions // max_frames
                    )

                    frame_number = 0

                    for y in range(result_h):

                        for x in range(result_w):

                            if frame_number % step == 0:

                                scan_frame = (
                                    main_array.copy()
                                )

                                cv2.rectangle(
                                    scan_frame,
                                    (x, y),
                                    (
                                        x + template_width,
                                        y + template_height
                                    ),
                                    (0, 255, 255),
                                    3
                                )

                                scan_placeholder.image(
                                    scan_frame,
                                    caption=(
                                        f"Scanning "
                                        f"X={x}, Y={y}"
                                    ),
                                    width=500
                                )

                                time.sleep(0.015)

                            frame_number += 1

                    min_val, max_val, min_loc, max_loc = (
                        cv2.minMaxLoc(result)
                    )

                    best_x, best_y = max_loc

                    matched_image = (
                        main_array.copy()
                    )

                    cv2.rectangle(
                        matched_image,
                        (best_x, best_y),
                        (
                            best_x + template_width,
                            best_y + template_height
                        ),
                        (0, 255, 255),
                        4
                    )

                    matched_image_bgr = cv2.cvtColor(
                        matched_image,
                        cv2.COLOR_RGB2BGR
                    )

                    cv2.putText(
                        matched_image_bgr,
                        "Perfect Match",
                        (
                            best_x,
                            max(best_y - 10, 25)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    matched_image = cv2.cvtColor(
                        matched_image_bgr,
                        cv2.COLOR_BGR2RGB
                    )

                    st.session_state.tm_result = {
                        "image": matched_image,
                        "score": float(max_val),
                        "x": best_x,
                        "y": best_y,
                        "width": template_width,
                        "height": template_height
                    }

                    st.session_state.tm_data = {
                        "main": main_array,
                        "template": template_array,
                        "result": result,
                        "best_x": best_x,
                        "best_y": best_y,
                        "template_width": template_width,
                        "template_height": template_height
                    }

                    st.session_state.tm_scan_done = True

                    st.rerun()

        if st.session_state.tm_result is not None:

            result_data = (
                st.session_state.tm_result
            )

            st.divider()

            st.subheader("🎯 Perfect Match")

            st.image(
                result_data["image"],
                caption="Perfect Match",
                width=500
            )

            result_info_1, result_info_2, result_info_3 = (
                st.columns(3)
            )

            with result_info_1:

                st.metric(
                    "Similarity",
                    f"{result_data['score']:.2f}"
                )

            with result_info_2:

                st.metric(
                    "X",
                    result_data["x"]
                )

            with result_info_3:

                st.metric(
                    "Y",
                    result_data["y"]
                )

            st.caption(
                f"Region Size: "
                f"{result_data['width']} × "
                f"{result_data['height']}"
            )

    with right:

        st.subheader("⚙️ How Template Matching Works")

        flow = st.columns(7)

        with flow[0]:
            st.write("📷")
            st.caption("Main Image")

        with flow[1]:
            st.write("→")

        with flow[2]:
            st.write("🔲")
            st.caption("Template")

        with flow[3]:
            st.write("→")

        with flow[4]:
            st.write("🔍")
            st.caption("Scan")

        with flow[5]:
            st.write("→")

        with flow[6]:
            st.write("🧮")
            st.caption("Compare")

        flow2 = st.columns(5)

        with flow2[0]:
            st.write("📊")
            st.caption("Similarity")

        with flow2[1]:
            st.write("→")

        with flow2[2]:
            st.write("🏆")
            st.caption("Best Score")

        with flow2[3]:
            st.write("→")

        with flow2[4]:
            st.write("🎯")
            st.caption("Match")

        st.divider()

        st.subheader("🧠 Explanation")

        st.write(
            "**1. Main Image** — The large image in which "
            "the template will be searched."
        )

        st.write(
            "**2. Template** — A smaller image that we want "
            "to find inside the main image."
        )

        st.write(
            "**3. Scan** — The template slides across every "
            "valid position of the main image."
        )

        st.write(
            "**4. Compare** — Each image region is compared "
            "with the template."
        )

        st.write(
            "**5. Similarity** — OpenCV calculates a similarity "
            "score for every position."
        )

        st.write(
            "**6. Best Match** — The position with the highest "
            "similarity score becomes the final match."
        )


# =========================================================
# VIOLA–JONES
# =========================================================

def show_viola_jones():

    if st.button(
        "← Back to Techniques",
        key="vj_back"
    ):

        st.session_state.page = "home"
        st.session_state.vj_result = None

        st.rerun()

    st.title("Viola–Jones")

    st.write(
        "Detect faces using Haar-like features, "
        "integral images, AdaBoost and cascade classifiers."
    )

    st.divider()

    left, right = st.columns(
        [1, 1.15],
        gap="medium"
    )

    with left:

        st.subheader("📷 Face Detection")

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            key="vj_upload"
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            image_array = np.array(image)

            st.image(
                image_array,
                caption="Input Image",
                width=420
            )

            find_face = st.button(
                "🔍 Find Face",
                key="find_face",
                use_container_width=True
            )

            if find_face:

                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                gray = cv2.cvtColor(
                    image_bgr,
                    cv2.COLOR_BGR2GRAY
                )

                cascade_path = (
                    cv2.data.haarcascades
                    + "haarcascade_frontalface_default.xml"
                )

                face_cascade = cv2.CascadeClassifier(
                    cascade_path
                )

                if face_cascade.empty():

                    st.error(
                        "Could not load the Viola–Jones "
                        "Haar Cascade classifier."
                    )

                else:

                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    result = image_bgr.copy()

                    for (x, y, w, h) in faces:

                        cv2.rectangle(
                            result,
                            (x, y),
                            (x + w, y + h),
                            (255, 200, 50),
                            3
                        )

                        cv2.putText(
                            result,
                            "Face",
                            (
                                x,
                                max(y - 10, 20)
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 200, 50),
                            2
                        )

                    result_rgb = cv2.cvtColor(
                        result,
                        cv2.COLOR_BGR2RGB
                    )

                    st.session_state.vj_result = {
                        "image": result_rgb,
                        "count": len(faces)
                    }

        if st.session_state.vj_result is not None:

            result_data = (
                st.session_state.vj_result
            )

            st.divider()

            st.subheader("🎯 Detection Result")

            st.image(
                result_data["image"],
                caption="Detected Faces",
                width=420
            )

            count = result_data["count"]

            if count == 0:

                st.warning(
                    "No face detected."
                )

            elif count == 1:

                st.success(
                    "✓ 1 Face Detected"
                )

            else:

                st.success(
                    f"✓ {count} Faces Detected"
                )

    with right:

        st.subheader("⚙️ How Viola–Jones Works")

        flow = st.columns(7)

        with flow[0]:
            st.write("📷")
            st.caption("Input")

        with flow[1]:
            st.write("→")

        with flow[2]:
            st.write("⚫")
            st.caption("Gray")

        with flow[3]:
            st.write("→")

        with flow[4]:
            st.write("⬛")
            st.caption("Haar")

        with flow[5]:
            st.write("→")

        with flow[6]:
            st.write("🧮")
            st.caption("Integral")

        flow2 = st.columns(5)

        with flow2[0]:
            st.write("🤖")
            st.caption("AdaBoost")

        with flow2[1]:
            st.write("→")

        with flow2[2]:
            st.write("🔗")
            st.caption("Cascade")

        with flow2[3]:
            st.write("→")

        with flow2[4]:
            st.write("🙂")
            st.caption("Face")


# =========================================================
# DEEPFACE
# =========================================================

def show_deepface():

    try:

        from deepface import DeepFace

    except Exception as e:

        st.error(
            "DeepFace could not be loaded."
        )

        st.code(
            str(e)
        )

        st.info(
            'Run:\n\n'
            'python -m pip install --upgrade "deepface[tensorflow]"'
        )

        return

    if st.button(
        "← Back to Techniques",
        key="df_back"
    ):

        st.session_state.page = "home"
        st.session_state.df_result = None

        st.rerun()

    st.title("🤖 DeepFace")

    st.write(
        "Compare two face images and determine whether "
        "they belong to the same person."
    )

    st.divider()

    left, right = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with left:

        st.subheader("📷 Face Verification")

        image1_file = st.file_uploader(
            "Upload Image 1",
            type=["jpg", "jpeg", "png"],
            key="df_image1"
        )

        image2_file = st.file_uploader(
            "Upload Image 2",
            type=["jpg", "jpeg", "png"],
            key="df_image2"
        )

        if (
            image1_file is not None
            and image2_file is not None
        ):

            image1 = Image.open(
                image1_file
            ).convert("RGB")

            image2 = Image.open(
                image2_file
            ).convert("RGB")

            image1_rgb = np.array(image1)
            image2_rgb = np.array(image2)

            image1_bgr = cv2.cvtColor(
                image1_rgb,
                cv2.COLOR_RGB2BGR
            )

            image2_bgr = cv2.cvtColor(
                image2_rgb,
                cv2.COLOR_RGB2BGR
            )

            img_col1, img_col2 = st.columns(2)

            with img_col1:

                st.image(
                    image1_rgb,
                    caption="Image 1",
                    width=280
                )

            with img_col2:

                st.image(
                    image2_rgb,
                    caption="Image 2",
                    width=280
                )

            st.write("")

            verify_button = st.button(
                "🔍 Verify Faces",
                key="verify_faces",
                use_container_width=True
            )

            if verify_button:

                try:

                    with st.spinner(
                        "DeepFace is comparing the faces..."
                    ):

                        verification = DeepFace.verify(
                            img1_path=image1_bgr,
                            img2_path=image2_bgr,
                            model_name="VGG-Face",
                            detector_backend="opencv",
                            distance_metric="cosine",
                            enforce_detection=True,
                            align=True,
                            silent=True
                        )

                    st.session_state.df_result = {
                        "verified": bool(
                            verification["verified"]
                        ),
                        "distance": float(
                            verification["distance"]
                        ),
                        "threshold": float(
                            verification["threshold"]
                        ),
                        "confidence": float(
                            verification.get(
                                "confidence",
                                0
                            )
                        ),
                        "model": verification.get(
                            "model",
                            "VGG-Face"
                        ),
                        "metric": verification.get(
                            "distance_metric",
                            "cosine"
                        )
                    }

                except Exception as e:

                    st.session_state.df_result = {
                        "error": str(e)
                    }

            if st.session_state.df_result is not None:

                result = st.session_state.df_result

                st.divider()

                st.subheader("🎯 Verification Result")

                if "error" in result:

                    st.error(
                        "❌ Could not verify the faces."
                    )

                    st.caption(
                        "Make sure both images contain "
                        "a clearly visible face."
                    )

                    st.code(
                        result["error"]
                    )

                elif result["verified"]:

                    st.success(
                        "✅ Match Detected"
                    )

                    st.write(
                        "The two images appear to be "
                        "the same person."
                    )

                else:

                    st.error(
                        "❌ No Match"
                    )

                    st.write(
                        "The two images appear to be "
                        "different people."
                    )

                if "error" not in result:

                    metric_col1, metric_col2, metric_col3 = (
                        st.columns(3)
                    )

                    with metric_col1:

                        st.metric(
                            "Distance",
                            f"{result['distance']:.4f}"
                        )

                    with metric_col2:

                        st.metric(
                            "Threshold",
                            f"{result['threshold']:.4f}"
                        )

                    with metric_col3:

                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.1f}%"
                        )

                    st.caption(
                        f"Model: {result['model']}  |  "
                        f"Metric: {result['metric']}"
                    )

    with right:

        st.subheader(
            "⚙️ How DeepFace Verification Works"
        )

        flow = st.columns(7)

        with flow[0]:
            st.write("📷")
            st.caption("Image 1")

        with flow[1]:
            st.write("+")

        with flow[2]:
            st.write("📷")
            st.caption("Image 2")

        with flow[3]:
            st.write("→")

        with flow[4]:
            st.write("🙂")
            st.caption("Faces")

        with flow[5]:
            st.write("→")

        with flow[6]:
            st.write("🧠")
            st.caption("Model")

        flow2 = st.columns(5)

        with flow2[0]:
            st.write("🔢")
            st.caption("Embeddings")

        with flow2[1]:
            st.write("→")

        with flow2[2]:
            st.write("📊")
            st.caption("Distance")

        with flow2[3]:
            st.write("→")

        with flow2[4]:
            st.write("🎯")
            st.caption("Match?")

        st.divider()

        st.subheader("🧠 Explanation")

        st.write(
            "**1. Two Images** — Upload two images containing "
            "faces that you want to compare."
        )

        st.write(
            "**2. Face Detection** — DeepFace detects the face "
            "in each image."
        )

        st.write(
            "**3. Face Embeddings** — The detected faces are "
            "converted into numerical representations."
        )

        st.write(
            "**4. Comparison** — DeepFace compares the two "
            "face representations using a distance metric."
        )

        st.write(
            "**5. Threshold** — The calculated distance is "
            "compared with a model-specific threshold."
        )

        st.write(
            "**6. Final Result** — If the distance is within "
            "the threshold, a match is detected."
        )


# =========================================================
# FACENET
# =========================================================

def show_facenet():

    try:

        from deepface import DeepFace

    except Exception as e:

        st.error(
            "FaceNet could not be loaded because DeepFace "
            "could not be imported."
        )

        st.code(
            str(e)
        )

        st.info(
            'Run:\n\n'
            'python -m pip install --upgrade "deepface[tensorflow]"'
        )

        return

    if st.button(
        "← Back to Techniques",
        key="fn_back"
    ):

        st.session_state.page = "home"
        st.session_state.fn_result = None

        st.rerun()

    st.title("🔗 FaceNet")

    st.write(
        "Convert a face image into a numerical embedding "
        "using the FaceNet deep-learning model."
    )

    st.divider()

    left, right = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with left:

        st.subheader("📷 Face Input")

        uploaded_file = st.file_uploader(
            "Upload Face Image",
            type=["jpg", "jpeg", "png"],
            key="fn_upload"
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            image_rgb = np.array(image)

            st.image(
                image_rgb,
                caption="Input Face",
                width=450
            )

            st.write("")

            generate_embedding = st.button(
                "🔗 Generate Face Embedding",
                key="generate_embedding",
                use_container_width=True
            )

            if generate_embedding:

                try:

                    image_bgr = cv2.cvtColor(
                        image_rgb,
                        cv2.COLOR_RGB2BGR
                    )

                    with st.spinner(
                        "FaceNet is generating the embedding..."
                    ):

                        embedding_result = (
                            DeepFace.represent(
                                img_path=image_bgr,
                                model_name="Facenet",
                                detector_backend="opencv",
                                enforce_detection=True,
                                align=True
                            )
                        )

                    if isinstance(
                        embedding_result,
                        list
                    ):

                        embedding = np.array(
                            embedding_result[0]["embedding"],
                            dtype=float
                        )

                    else:

                        embedding = np.array(
                            embedding_result["embedding"],
                            dtype=float
                        )

                    st.session_state.fn_result = {
                        "embedding": embedding
                    }

                except Exception as e:

                    st.session_state.fn_result = {
                        "error": str(e)
                    }

            if st.session_state.fn_result is not None:

                result = st.session_state.fn_result

                st.divider()

                st.subheader("🎯 FaceNet Result")

                if "error" in result:

                    st.error(
                        "❌ Could not generate the face embedding."
                    )

                    st.caption(
                        "Make sure the uploaded image contains "
                        "a clearly visible face."
                    )

                    st.code(
                        result["error"]
                    )

                else:

                    embedding = result["embedding"]

                    st.success(
                        "✅ Face Embedding Generated"
                    )

                    st.write(
                        f"FaceNet converted the detected face "
                        f"into a **{len(embedding)}-dimensional "
                        f"numerical vector**."
                    )

                    metric1, metric2 = st.columns(2)

                    with metric1:

                        st.metric(
                            "Embedding Dimensions",
                            len(embedding)
                        )

                    with metric2:

                        st.metric(
                            "Vector Magnitude",
                            f"{np.linalg.norm(embedding):.4f}"
                        )

                    st.write("")

                    st.subheader(
                        "🔢 Embedding Values"
                    )

                    st.caption(
                        "First 20 values of the FaceNet vector"
                    )

                    embedding_preview = np.round(
                        embedding[:20],
                        4
                    )

                    st.code(
                        str(
                            embedding_preview.tolist()
                        )
                    )

    with right:

        st.subheader(
            "⚙️ How FaceNet Works"
        )

        flow = st.columns(7)

        with flow[0]:

            st.write("📷")
            st.caption("Image")

        with flow[1]:

            st.write("→")

        with flow[2]:

            st.write("🙂")
            st.caption("Face")

        with flow[3]:

            st.write("→")

        with flow[4]:

            st.write("🧠")
            st.caption("FaceNet")

        with flow[5]:

            st.write("→")

        with flow[6]:

            st.write("🔢")
            st.caption("Vector")

        flow2 = st.columns(5)

        with flow2[0]:

            st.write("📊")
            st.caption("Features")

        with flow2[1]:

            st.write("→")

        with flow2[2]:

            st.write("🔢")
            st.caption("Embedding")

        with flow2[3]:

            st.write("→")

        with flow2[4]:

            st.write("🎯")
            st.caption("Identity")

        st.divider()

        st.subheader("🧠 Explanation")

        st.write(
            "**1. Input Image** — Upload an image containing "
            "a clearly visible face."
        )

        st.write(
            "**2. Face Detection** — The face is detected "
            "before it is processed by FaceNet."
        )

        st.write(
            "**3. Feature Extraction** — FaceNet's deep "
            "neural network extracts important facial features."
        )

        st.write(
            "**4. Embedding** — The facial features are "
            "converted into a numerical vector."
        )

        st.write(
            "**5. Face Representation** — The resulting "
            "embedding represents the characteristics of "
            "that face."
        )

        st.write(
            "**6. Recognition** — Face embeddings can later "
            "be compared to recognize or verify identities."
        )


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "home":

    show_home()

elif st.session_state.page == "template_matching":

    show_template_matching()

elif st.session_state.page == "viola_jones":

    show_viola_jones()

elif st.session_state.page == "deepface":

    show_deepface()

elif st.session_state.page == "facenet":

    show_facenet()