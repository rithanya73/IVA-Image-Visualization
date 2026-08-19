from flask import Flask, request, render_template_string
import webbrowser
from threading import Timer
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IVA - Image Visualization</title>

    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #222;
        }

        .header {
            background: #202124;
            color: white;
            padding: 25px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 30px;
        }

        .header p {
            margin-top: 8px;
            color: #ccc;
        }

        .container {
            width: 90%;
            max-width: 1100px;
            margin: 30px auto;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.1);
        }

        h2 {
            margin-top: 0;
        }

        input[type="file"] {
            margin: 15px 0;
        }

        select, button {
            padding: 11px 15px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 15px;
        }

        button {
            background: #202124;
            color: white;
            cursor: pointer;
            border: none;
        }

        button:hover {
            background: #444;
        }

        .images {
            display: flex;
            gap: 25px;
            flex-wrap: wrap;
        }

        .image-box {
            flex: 1;
            min-width: 300px;
            text-align: center;
        }

        .image-box img {
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }

        .info {
            background: #f1f3f4;
            padding: 15px;
            border-radius: 8px;
            line-height: 1.6;
        }

        .operator {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .operator div {
            padding: 18px;
            background: #f1f3f4;
            border-radius: 8px;
        }
    </style>
</head>

<body>

<div class="header">
    <h1>Image Visualization & Analysis</h1>
    <p>Spatial Domain Method - Gradient Operators</p>
</div>

<div class="container">

    <div class="card">

        <h2>Upload Image</h2>

        <form method="POST" enctype="multipart/form-data">

            <input type="file" name="image" accept="image/*" required>

            <br><br>

            <label><b>Select Gradient Operator:</b></label>

            <select name="operator">
                <option value="sobel">Sobel Operator</option>
                <option value="prewitt">Prewitt Operator</option>
                <option value="roberts">Roberts Operator</option>
                <option value="laplacian">Laplacian Operator</option>
            </select>

            <br><br>

            <button type="submit">Process Image</button>

        </form>

    </div>

    {% if original %}

    <div class="card">

        <h2>Results</h2>

        <div class="images">

            <div class="image-box">
                <h3>Original Image</h3>
                <img src="data:image/png;base64,{{ original }}">
            </div>

            <div class="image-box">
                <h3>{{ operator_name }}</h3>
                <img src="data:image/png;base64,{{ result }}">
            </div>

        </div>

    </div>

    <div class="card">

        <h2>Gradient Operator Information</h2>

        <div class="info">

            {% if operator == "sobel" %}

            <b>Sobel Operator</b>
            <p>
                Sobel operator detects edges by calculating the
                horizontal and vertical intensity changes in an image.
            </p>

            <p>
                It uses two 3 × 3 kernels for detecting X and Y
                directions.
            </p>

            {% elif operator == "prewitt" %}

            <b>Prewitt Operator</b>
            <p>
                Prewitt operator is used for detecting edges and
                boundaries in an image.
            </p>

            <p>
                It uses horizontal and vertical 3 × 3 kernels.
            </p>

            {% elif operator == "roberts" %}

            <b>Roberts Operator</b>
            <p>
                Roberts operator is a simple gradient operator that
                detects edges using 2 × 2 kernels.
            </p>

            {% elif operator == "laplacian" %}

            <b>Laplacian Operator</b>
            <p>
                Laplacian is a second-order derivative operator.
                It detects regions of rapid intensity change.
            </p>

            {% endif %}

        </div>

    </div>

    {% endif %}

</div>

</body>
</html>
"""


def image_to_base64(image):
    """
    Convert OpenCV image to Base64 so that
    it can be displayed directly in the browser.
    """

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def apply_gradient(image, operator):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---------------- SOBEL ----------------

    if operator == "sobel":

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            sobel_x.astype(np.float32),
            sobel_y.astype(np.float32)
        )

        result = cv2.convertScaleAbs(magnitude)

        return result

    # ---------------- PREWITT ----------------

    elif operator == "prewitt":

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

        result = cv2.convertScaleAbs(magnitude)

        return result

    # ---------------- ROBERTS ----------------

    elif operator == "roberts":

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

        result = cv2.convertScaleAbs(magnitude)

        return result

    # ---------------- LAPLACIAN ----------------

    elif operator == "laplacian":

        result = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        result = cv2.convertScaleAbs(result)

        return result

    return gray


@app.route("/", methods=["GET", "POST"])
def home():

    original = None
    result = None
    operator = None
    operator_name = None

    if request.method == "POST":

        file = request.files.get("image")

        operator = request.form.get(
            "operator",
            "sobel"
        )

        if file:

            file_bytes = np.frombuffer(
                file.read(),
                np.uint8
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

                original = image_to_base64(
                    image
                )

                result = image_to_base64(
                    processed
                )

                names = {
                    "sobel": "Sobel Edge Detection",
                    "prewitt": "Prewitt Edge Detection",
                    "roberts": "Roberts Edge Detection",
                    "laplacian": "Laplacian Edge Detection"
                }

                operator_name = names.get(
                    operator,
                    "Gradient Result"
                )

    return render_template_string(
        HTML,
        original=original,
        result=result,
        operator=operator,
        operator_name=operator_name
    )
if __name__ == "__main__":

    print("--------------------------------------")
    print("IVA Image Visualization Application")
    print("--------------------------------------")
    print("Opening browser...")
    print("--------------------------------------")

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    Timer(1, open_browser).start()

    app.run(debug=True, use_reloader=False)