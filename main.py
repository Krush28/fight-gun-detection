import os
import threading
import time

import cv2
import imutils
import torch
from flask import Flask, request, Response, jsonify
import dotenv

import fight_module


# ==========================================
# LOAD CONFIGURATION
# ==========================================

dotenv.load_dotenv()

YOLO_MODEL = os.getenv("YOLO_MODEL")
FIGHT_MODEL = os.getenv("FIGHT_MODEL")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# GLOBAL VARIABLES
# ==========================================

outputFrame = None
lock = threading.Lock()

fight_status = False
processing = False

app = Flask(__name__)


# ==========================================
# MAIN WEB PAGE
# ==========================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>

    <head>

        <title>Fight Detection System</title>

        <style>

            body {
                margin: 0;
                padding: 0;
                background: #111827;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }

            .container {
                width: 90%;
                max-width: 1100px;
                margin: 30px auto;
            }

            h1 {
                margin-bottom: 25px;
            }

            .controls {
                background: #1f2937;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
            }

            button {
                padding: 12px 22px;
                margin: 8px;
                border: none;
                border-radius: 8px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            input {
                margin: 10px;
                padding: 10px;
            }

            .feed {
                background: black;
                border-radius: 12px;
                overflow: hidden;
                min-height: 500px;
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .feed img {
                width: 100%;
                max-height: 700px;
                object-fit: contain;
            }

            #status {
                margin-top: 20px;
                padding: 18px;
                border-radius: 10px;
                background: #374151;
                font-size: 24px;
                font-weight: bold;
            }

            .normal {
                background: #065f46 !important;
            }

            .fight {
                background: #991b1b !important;
                animation: blink 0.7s infinite;
            }

            @keyframes blink {
                50% {
                    opacity: 0.5;
                }
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>🚨 Fight Detection System</h1>

            <div class="controls">

                <h2>Live Camera</h2>

                <button onclick="startCamera()">
                    📹 Start Live Camera
                </button>

                <button onclick="stopDetection()">
                    ⛔ Stop
                </button>

                <hr>

                <h2>Upload Video</h2>

                <input
                    type="file"
                    id="videoFile"
                    accept="video/*"
                >

                <button onclick="uploadVideo()">
                    ▶ Start Fight Detection
                </button>

            </div>


            <div class="feed">

                <img
                    id="videoFeed"
                    src=""
                    alt="Detection feed"
                >

            </div>


            <div id="status">
                ⚪ Waiting for detection...
            </div>

        </div>


        <script>

            let statusInterval = null;


            function startCamera() {

                fetch("/start_camera")
                    .then(response => response.json())
                    .then(data => {

                        if (data.success) {

                            document.getElementById("videoFeed").src =
                                "/raw?" + Date.now();

                            startStatusChecking();

                        }

                    });

            }


            function uploadVideo() {

                const fileInput =
                    document.getElementById("videoFile");

                if (!fileInput.files.length) {

                    alert("Please select a video first.");

                    return;

                }


                const formData = new FormData();

                formData.append(
                    "video",
                    fileInput.files[0]
                );


                fetch("/upload", {

                    method: "POST",
                    body: formData

                })

                .then(response => response.json())

                .then(data => {

                    if (data.success) {

                        document.getElementById("videoFeed").src =
                            "/raw?" + Date.now();

                        startStatusChecking();

                    }
                    else {

                        alert(data.message);

                    }

                })

                .catch(error => {

                    console.error(error);

                    alert("Upload failed.");

                });

            }


            function startStatusChecking() {

                if (statusInterval) {

                    clearInterval(statusInterval);

                }


                statusInterval = setInterval(() => {

                    fetch("/status")

                        .then(response => response.json())

                        .then(data => {

                            const status =
                                document.getElementById("status");


                            if (data.fight_detected) {

                                status.innerHTML =
                                    "🔴 🚨 FIGHT DETECTED 🚨";

                                status.className = "fight";

                            }
                            else if (data.processing) {

                                status.innerHTML =
                                    "🟢 Monitoring... No Fight Detected";

                                status.className = "normal";

                            }
                            else {

                                status.innerHTML =
                                    "⚪ Waiting for detection...";

                                status.className = "";

                            }

                        });

                }, 500);

            }


            function stopDetection() {

                fetch("/stop");

                document.getElementById("videoFeed").src = "";

                document.getElementById("status").innerHTML =
                    "⚪ Detection stopped";

                document.getElementById("status").className = "";

            }

        </script>

    </body>

    </html>
    """


# ==========================================
# NVIDIA / GPU INFORMATION
# ==========================================

@app.route("/nvidia")
def nvidia():

    gpu_info = ""

    if torch.cuda.is_available():

        gpu_info += "CUDA is available.\n"

        for i in range(torch.cuda.device_count()):

            gpu = torch.cuda.get_device_properties(i)

            gpu_info += (
                f"GPU {i}: {gpu.name}\n"
            )

    else:

        gpu_info = "CUDA is not available."

    return gpu_info


# ==========================================
# VIDEO STREAM
# ==========================================

@app.route("/raw")
def raw():

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ==========================================
# UPLOAD VIDEO
# ==========================================

@app.route("/upload", methods=["POST"])
def upload():

    global processing
    global fight_status
    global outputFrame

    if "video" not in request.files:

        return jsonify({
            "success": False,
            "message": "No video uploaded."
        })


    video = request.files["video"]

    if video.filename == "":

        return jsonify({
            "success": False,
            "message": "No video selected."
        })


    filename = "uploaded_video.mp4"

    video_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    video.save(video_path)


    # Reset previous detection
    fight_status = False
    processing = True
    outputFrame = None


    # Start detection thread
    t = threading.Thread(
        target=detect,
        args=(video_path,)
    )

    t.daemon = True
    t.start()


    return jsonify({
        "success": True,
        "message": "Video uploaded. Detection started."
    })


# ==========================================
# START LIVE CAMERA
# ==========================================

@app.route("/start_camera")
def start_camera():

    global processing
    global fight_status
    global outputFrame

    fight_status = False
    processing = True
    outputFrame = None


    t = threading.Thread(
        target=detect,
        args=(0,)
    )

    t.daemon = True
    t.start()


    return jsonify({
        "success": True
    })


# ==========================================
# STATUS
# ==========================================

@app.route("/status")
def status():

    return jsonify({
        "fight_detected": fight_status,
        "processing": processing
    })


# ==========================================
# STOP
# ==========================================

@app.route("/stop")
def stop():

    global processing
    global fight_status

    processing = False
    fight_status = False

    return jsonify({
        "success": True
    })


# ==========================================
# FIGHT DETECTION
# ==========================================

def detect(video_input):

    global outputFrame
    global fight_status
    global processing


    FIGHT_ON_TIMEOUT = 0


    try:

        fdet = fight_module.FightDetector(
            FIGHT_MODEL
        )

        yolo = fight_module.YoloPoseEstimation(
            YOLO_MODEL
        )


        for result in yolo.estimate(video_input):

            if not processing:

                break


            result_frame = result.plot()


            if result_frame.shape[0] > 720:

                result_frame = imutils.resize(
                    result_frame,
                    width=1280
                )


            current_fight = False


            try:

                boxes = result.boxes.xyxy.tolist()

                xyn = result.keypoints.xyn.tolist()

                confs = result.keypoints.conf

                ids = result.boxes.id


                confs = (
                    []
                    if confs is None
                    else confs.tolist()
                )


                ids = (
                    []
                    if ids is None
                    else [
                        str(int(ID))
                        for ID in ids
                    ]
                )


                interaction_boxes = (
                    fight_module.get_interaction_box(
                        boxes
                    )
                )


                for inter_box in interaction_boxes:


                    cv2.rectangle(

                        result_frame,

                        (
                            int(inter_box[0]),
                            int(inter_box[1])
                        ),

                        (
                            int(inter_box[2]),
                            int(inter_box[3])
                        ),

                        (0, 255, 0),

                        2
                    )


                    both_fighting = []


                    for conf, pose, box, identity in zip(
                        confs,
                        xyn,
                        boxes,
                        ids
                    ):


                        center_person_x = (
                            box[2] + box[0]
                        ) / 2


                        center_person_y = (
                            box[3] + box[1]
                        ) / 2


                        if (

                            inter_box[0]
                            <= center_person_x
                            <= inter_box[2]

                            and

                            inter_box[1]
                            <= center_person_y
                            <= inter_box[3]

                        ):


                            is_person_fighting = (
                                fdet.detect(
                                    conf,
                                    pose
                                )
                            )


                            both_fighting.append(
                                is_person_fighting
                            )


                    if (
                        len(both_fighting) >= 2
                        and all(both_fighting)
                    ):

                        current_fight = True


            except (TypeError, IndexError):

                pass


            # ==================================
            # FIGHT STATUS
            # ==================================

            if current_fight:

                fight_status = True

                FIGHT_ON_TIMEOUT = 30


            elif FIGHT_ON_TIMEOUT > 0:

                FIGHT_ON_TIMEOUT -= 1

            else:

                fight_status = False


            # ==================================
            # DRAW ALERT ON VIDEO
            # ==================================

            if fight_status:

                cv2.rectangle(

                    result_frame,

                    (0, 0),

                    (
                        result_frame.shape[1],
                        70
                    ),

                    (0, 0, 255),

                    -1
                )


                cv2.putText(

                    result_frame,

                    "!!! FIGHT DETECTED !!!",

                    (30, 48),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.3,

                    (255, 255, 255),

                    3,

                    cv2.LINE_AA
                )


            else:

                cv2.putText(

                    result_frame,

                    "MONITORING - NO FIGHT",

                    (30, 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.9,

                    (0, 255, 0),

                    2,

                    cv2.LINE_AA
                )


            # ==================================
            # SAVE FRAME FOR BROWSER
            # ==================================

            with lock:

                outputFrame = result_frame.copy()


    except Exception as e:

        print("Detection error:", e)


    finally:

        processing = False

        fight_status = False


# ==========================================
# GENERATE MJPEG STREAM
# ==========================================

def generate():

    global outputFrame
    global lock


    while True:

        if outputFrame is None:

            time.sleep(0.01)

            continue


        with lock:

            frame = outputFrame.copy()


        flag, encodedImage = cv2.imencode(
            ".jpg",
            frame
        )


        if not flag:

            continue


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + bytearray(encodedImage)
            + b"\r\n"
        )


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=8000,

        debug=True,

        threaded=True,

        use_reloader=False
    )