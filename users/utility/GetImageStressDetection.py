import cv2 as cv
from PyEmotion import DetectFace, PyEmotion
from collections import Counter
import time
from django.conf import settings


class ImageExpressionDetect:
    def __init__(self):
        self.detected_emotions = []

    def getExpression(self,imagepath):
        filepath = settings.MEDIA_ROOT + "\\" + imagepath
        PyEmotion()
        er = DetectFace(device='cpu', gpu_id=0)
        frame, emotion = er.predict_emotion(cv.imread(filepath))
        cv.imshow('Alex Corporation', frame)
        cv.waitKey(0)
        print("Hola Hi",filepath,"Emotion is ",emotion)
        return emotion

    def getLiveDetect(self):
        print("Streaming Started")
        PyEmotion()
        er = DetectFace(device='cpu', gpu_id=0)
        cap = cv.VideoCapture(0)
        start_time = time.time()
        last_alert_time = start_time

        if not cap.isOpened():
            print("Error: Failed to open camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            frame, emotion = er.predict_emotion(frame)
            self.detected_emotions.append(emotion)
            cv.imshow('Press Q to Exit', frame)
            current_time = time.time()

            if current_time - last_alert_time >= 30:
                most_common_emotion = self.get_most_common_emotion()
                if most_common_emotion and most_common_emotion != "NoFace":
                    print("Most common emotion detected:", most_common_emotion)
                    self.display_alert("Most common emotion detected: " + most_common_emotion)
                else:
                    # If most common emotion is "NoFace", get the second most common emotion
                    second_most_common_emotion = self.get_second_most_common_emotion()
                    if second_most_common_emotion:
                        print("Second most common emotion detected:", second_most_common_emotion)
                        self.display_alert("Second most common emotion detected: " + second_most_common_emotion)
                    else:
                        print("No valid emotions detected.")
                last_alert_time = current_time

            key = cv.waitKey(1)  # Update the OpenCV window
            if key == ord('q'):
                print("Exiting live detection")
                break

        cap.release()
        cv.destroyAllWindows()

    def display_alert(self, message):
        # Function to display alert message
        print("Alert:", message)
        # Replace print with the code to display alert box using JavaScript in a web page

    def get_most_common_emotion(self):
        if self.detected_emotions:
            emotion_counter = Counter(self.detected_emotions)
            # Exclude "NoFace" from the calculation
            emotion_counter.pop("NoFace", None)
            if emotion_counter:
                most_common_emotion, _ = emotion_counter.most_common(1)[0]
                return most_common_emotion
            else:
                return None
        else:
            return None

    def get_second_most_common_emotion(self):
        if self.detected_emotions:
            emotion_counter = Counter(self.detected_emotions)
            # Exclude "NoFace" from the calculation
            emotion_counter.pop("NoFace", None)
            if emotion_counter:
                most_common_emotion, _ = emotion_counter.most_common(2)[-1]
                return most_common_emotion
            else:
                return None
        else:
            return None

# Example usage:
if __name__ == "__main__":
    detector = ImageExpressionDetect()
    detector.getLiveDetect()
