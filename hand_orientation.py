import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


def get_orientation(landmarks, image_width, image_height):
    # Key points
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    middle_mcp = landmarks[9]

    def to_px(pt):
        return int(pt.x * image_width), int(pt.y * image_height)

    index_px = to_px(index_mcp)
    pinky_px = to_px(pinky_mcp)
    middle_px = to_px(middle_mcp)

    # Calculate angle between index and pinky base
    hand_vector_x = index_px[0] - pinky_px[0]
    hand_vector_y = index_px[1] - pinky_px[1]
    angle = math.degrees(math.atan2(hand_vector_y, hand_vector_x))

    # Calculate z difference (depth)
    z_diff = (index_mcp.z + pinky_mcp.z) / 2 - middle_mcp.z

    # Orientation classification based on your rule
    if -180 <= angle <= -150:
        orientation = "Straight"
    elif angle > 150:
        orientation = "Tilted Left"
    elif 0 < angle <= 150:
        orientation = "Left"

    elif angle > -150 and angle<-130:
        orientation = "Tilted Right"
    elif angle>-130:
        orientation = "Right"

    else:
        orientation = "Unknown"

    return orientation, angle


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = frame.shape

    orientation_text = "No Hand Detected"
    angle_val = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            orientation_text, angle_val = get_orientation(hand_landmarks.landmark, w, h)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display info on screen
    cv2.putText(frame, f"Orientation: {orientation_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Angle: {angle_val:.2f} deg", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    print(f"Angle: {angle_val:.2f}°  |  Orientation: {orientation_text}", end="\r")

    cv2.imshow("Hand Orientation Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
