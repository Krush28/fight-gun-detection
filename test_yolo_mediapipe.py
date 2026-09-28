import cv2
import time
import mediapipe as mp
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

YOLO_MODEL = "model/yolo/yolo11n.pt"
MEDIAPIPE_MODEL = "model/pose_landmarker_lite.task"

CAMERA_ID = 0

# YOLO confidence
YOLO_CONF = 0.5


# ============================================================
# COLORS
# ============================================================

PERSON_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 165, 255),    # Orange
    (255, 0, 255),    # Purple
    (0, 255, 255),    # Yellow
    (255, 255, 0),    # Cyan
]


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


# ============================================================
# GLOBAL MEDIAPIPE RESULT
# ============================================================

latest_result = None


def mediapipe_callback(
    result,
    output_image,
    timestamp_ms
):
    global latest_result

    latest_result = result


# ============================================================
# POSE CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [

    # Head
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Mouth
    (9, 10),

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Body
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28),

    # Feet
    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32),
]


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# CREATE MEDIAPIPE LANDMARKER
# ============================================================

print("Loading MediaPipe Pose...")

mp_options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MEDIAPIPE_MODEL
    ),

    running_mode=RunningMode.LIVE_STREAM,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    result_callback=mediapipe_callback
)


landmarker = PoseLandmarker.create_from_options(
    mp_options
)

print("MediaPipe loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    landmarker.close()

    exit()


print()
print("======================================")
print(" YOLO + BYTE TRACK + MEDIAPIPE")
print("======================================")
print("Press Q to quit.")
print()


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print("Could not read frame.")

        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # YOLO + BYTE TRACK
    # --------------------------------------------------------

    results = yolo.track(

        source=frame,

        persist=True,

        tracker="bytetrack.yaml",

        classes=[0],

        conf=YOLO_CONF,

        verbose=False
    )


    result = results[0]


    # --------------------------------------------------------
    # RESET MEDIAPIPE RESULT
    # --------------------------------------------------------

    latest_result = None


    # --------------------------------------------------------
    # PROCESS DETECTED PEOPLE
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes


        if boxes.id is not None:

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )


            xyxy_boxes = (
                boxes.xyxy
                .cpu()
                .tolist()
            )


            for person_index, (
                box,
                track_id
            ) in enumerate(
                zip(
                    xyxy_boxes,
                    track_ids
                )
            ):

                # ------------------------------------------------
                # PERSON BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # Keep coordinates inside frame

                x1 = max(
                    0,
                    min(x1, frame.shape[1] - 1)
                )

                x2 = max(
                    0,
                    min(x2, frame.shape[1] - 1)
                )

                y1 = max(
                    0,
                    min(y1, frame.shape[0] - 1)
                )

                y2 = max(
                    0,
                    min(y2, frame.shape[0] - 1)
                )


                if x2 <= x1 or y2 <= y1:

                    continue


                # ------------------------------------------------
                # COLOR FOR PERSON
                # ------------------------------------------------

                color = PERSON_COLORS[
                    int(track_id)
                    % len(PERSON_COLORS)
                ]


                # ------------------------------------------------
                # DRAW YOLO BOX
                # ------------------------------------------------

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    color,

                    2
                )


                # ------------------------------------------------
                # PERSON ID
                # ------------------------------------------------

                label = f"Person ID: {track_id}"


                cv2.rectangle(

                    frame,

                    (
                        x1,
                        max(0, y1 - 30)
                    ),

                    (
                        x1 + 160,
                        y1
                    ),

                    color,

                    -1
                )


                cv2.putText(

                    frame,

                    label,

                    (
                        x1 + 5,
                        max(20, y1 - 8)
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 0, 0),

                    2
                )


                # =================================================
                # CROP PERSON
                # =================================================

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if person_crop.size == 0:

                    continue


                # ------------------------------------------------
                # Convert crop to RGB
                # ------------------------------------------------

                rgb_crop = cv2.cvtColor(

                    person_crop,

                    cv2.COLOR_BGR2RGB
                )


                # ------------------------------------------------
                # MediaPipe Image
                # ------------------------------------------------

                mp_image = mp.Image(

                    image_format=mp.ImageFormat.SRGB,

                    data=rgb_crop
                )


                # ------------------------------------------------
                # MediaPipe
                # ------------------------------------------------

                timestamp_ms = int(
                    time.time() * 1000
                )


                landmarker.detect_async(

                    mp_image,

                    timestamp_ms
                )


                # ------------------------------------------------
                # Draw latest pose
                # ------------------------------------------------

                if latest_result is not None:

                    if (
                        len(
                            latest_result.pose_landmarks
                        ) > 0
                    ):

                        pose_landmarks = (
                            latest_result
                            .pose_landmarks[0]
                        )


                        # =========================================
                        # DRAW LANDMARKS
                        # =========================================

                        points = []


                        for landmark in pose_landmarks:

                            px = int(
                                landmark.x
                                * (x2 - x1)
                            )

                            py = int(
                                landmark.y
                                * (y2 - y1)
                            )


                            # Convert crop coordinates
                            # back to full-frame coordinates

                            px += x1
                            py += y1


                            points.append(
                                (px, py)
                            )


                        # =========================================
                        # DRAW CONNECTIONS
                        # =========================================

                        for start, end in POSE_CONNECTIONS:

                            if (
                                start < len(points)
                                and
                                end < len(points)
                            ):

                                p1 = points[start]

                                p2 = points[end]


                                cv2.line(

                                    frame,

                                    p1,

                                    p2,

                                    color,

                                    3
                                )


                        # =========================================
                        # DRAW LANDMARK POINTS
                        # =========================================

                        for point in points:

                            cv2.circle(

                                frame,

                                point,

                                5,

                                color,

                                -1
                            )


                        # =========================================
                        # POSE STATUS
                        # =========================================

                        cv2.putText(

                            frame,

                            "POSE DETECTED",

                            (
                                x1,
                                min(
                                    frame.shape[0] - 10,
                                    y2 + 25
                                )
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.55,

                            color,

                            2
                        )


    # ============================================================
    # FPS
    # ============================================================

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    # ============================================================
    # INFORMATION PANEL
    # ============================================================

    cv2.rectangle(

        frame,

        (10, 10),

        (350, 80),

        (20, 20, 20),

        -1
    )


    cv2.putText(

        frame,

        "YOLO + ByteTrack + MediaPipe",

        (20, 38),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )


    # ============================================================
    # SHOW
    # ============================================================

    cv2.imshow(

        "Security System - Tracking + Pose",

        frame
    )


    # ============================================================
    # QUIT
    # ============================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("System stopped.")