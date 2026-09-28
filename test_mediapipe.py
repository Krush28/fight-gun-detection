import cv2
import mediapipe as mp
import time

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/pose_landmarker_lite.task"

# Different colors for different people
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
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Store latest MediaPipe result
latest_result = None


# ============================================================
# CALLBACK
# ============================================================

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


# ============================================================
# POSE CONNECTIONS
# ============================================================

# MediaPipe Pose landmark connections
POSE_CONNECTIONS = [
    # Face / head
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

    # Torso
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
# MEDIAPIPE OPTIONS
# ============================================================

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.LIVE_STREAM,

    # Detect multiple people
    num_poses=4,

    # Detection confidence
    min_pose_detection_confidence=0.5,

    # Pose presence confidence
    min_pose_presence_confidence=0.5,

    # Tracking confidence
    min_tracking_confidence=0.5,

    # Callback for live stream
    result_callback=print_result
)


# ============================================================
# START MEDIAPIPE
# ============================================================

with PoseLandmarker.create_from_options(options) as landmarker:

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        exit()

    print("Camera started.")
    print("MediaPipe Pose started.")
    print("Press Q to quit.")

    # FPS calculation
    previous_time = 0

    while True:

        # ----------------------------------------------------
        # Read camera frame
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success:
            print("Failed to read frame.")
            break

        # Flip camera for mirror view
        frame = cv2.flip(frame, 1)

        # ----------------------------------------------------
        # Convert OpenCV → MediaPipe
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = int(time.time() * 1000)

        # ----------------------------------------------------
        # Run MediaPipe Pose
        # ----------------------------------------------------

        landmarker.detect_async(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # Draw results
        # ----------------------------------------------------

        if latest_result is not None:

            # Number of detected people
            number_of_people = len(
                latest_result.pose_landmarks
            )

            for person_id, pose_landmarks in enumerate(
                latest_result.pose_landmarks
            ):

                # ------------------------------------------------
                # Select color
                # ------------------------------------------------

                color = PERSON_COLORS[
                    person_id % len(PERSON_COLORS)
                ]

                # ------------------------------------------------
                # Convert landmarks to pixel coordinates
                # ------------------------------------------------

                points = []

                for landmark in pose_landmarks:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    points.append((x, y))

                # ------------------------------------------------
                # Calculate approximate person bounding box
                # ------------------------------------------------

                valid_points = []

                for landmark in pose_landmarks:

                    if (
                        0 <= landmark.x <= 1
                        and
                        0 <= landmark.y <= 1
                    ):
                        valid_points.append(
                            (
                                int(landmark.x * frame.shape[1]),
                                int(landmark.y * frame.shape[0])
                            )
                        )

                if len(valid_points) > 0:

                    x_values = [
                        point[0]
                        for point in valid_points
                    ]

                    y_values = [
                        point[1]
                        for point in valid_points
                    ]

                    min_x = max(
                        min(x_values) - 15,
                        0
                    )

                    max_x = min(
                        max(x_values) + 15,
                        frame.shape[1] - 1
                    )

                    min_y = max(
                        min(y_values) - 15,
                        0
                    )

                    max_y = min(
                        max(y_values) + 15,
                        frame.shape[0] - 1
                    )

                    # --------------------------------------------
                    # Draw bounding box
                    # --------------------------------------------

                    cv2.rectangle(
                        frame,
                        (min_x, min_y),
                        (max_x, max_y),
                        color,
                        2
                    )

                    # --------------------------------------------
                    # Person ID label
                    # --------------------------------------------

                    label = f"Person ID: {person_id}"

                    cv2.rectangle(
                        frame,
                        (
                            min_x,
                            max(min_y - 30, 0)
                        ),
                        (
                            min_x + 150,
                            min_y
                        ),
                        color,
                        -1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            min_x + 5,
                            min_y - 8
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2
                    )

                # ------------------------------------------------
                # Draw skeleton connections
                # ------------------------------------------------

                for start, end in POSE_CONNECTIONS:

                    if (
                        start < len(points)
                        and
                        end < len(points)
                    ):

                        x1, y1 = points[start]
                        x2, y2 = points[end]

                        # Only draw if coordinates are reasonable
                        if (
                            0 <= x1 < frame.shape[1]
                            and
                            0 <= y1 < frame.shape[0]
                            and
                            0 <= x2 < frame.shape[1]
                            and
                            0 <= y2 < frame.shape[0]
                        ):

                            cv2.line(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                color,
                                3
                            )

                # ------------------------------------------------
                # Draw landmark points
                # ------------------------------------------------

                for x, y in points:

                    if (
                        0 <= x < frame.shape[1]
                        and
                        0 <= y < frame.shape[0]
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            color,
                            -1
                        )

                        # Small black outline
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 0),
                            1
                        )

        # ====================================================
        # DISPLAY INFORMATION
        # ====================================================

        # FPS
        current_time = time.time()

        if previous_time != 0:

            fps = 1 / (
                current_time - previous_time
            )

        else:
            fps = 0

        previous_time = current_time

        # Number of people
        people_count = 0

        if latest_result is not None:
            people_count = len(
                latest_result.pose_landmarks
            )

        # ----------------------------------------------------
        # Top information panel
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (300, 85),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            "MediaPipe Pose",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"People: {people_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (170, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Press Q to quit",
            (
                20,
                frame.shape[0] - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW FRAME
        # ====================================================

        cv2.imshow(
            "Multimodal Security System - MediaPipe Pose",
            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()

print("Camera stopped.")