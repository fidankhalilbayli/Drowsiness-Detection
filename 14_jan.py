from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time 
import math


path_to_shape_predictor = r"C:\Users\fidan khalilbayli\Desktop\drowsiness/shape_predictor_68_face_landmarks.dat"
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_to_shape_predictor)
blink_counter = False
blink_flag=0
another_counter=0
some_counter=0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


cap = cv2.VideoCapture(0)

fps_int = math.ceil(cap.get(cv2.CAP_PROP_FPS))
time.sleep(1.0)
frame_counter  = 0
writer = None

def mouth_aspect_ratio(mouth):
	return np.mean(dist.euclidean(mouth[2], mouth[10])+dist.euclidean(mouth[4], mouth[8]))/dist.euclidean(mouth[0], mouth[6])

def get_gaze_ratio(eye_points, facial_landmarks):
	left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

	height, width, _ = frame.shape
	mask = np.zeros((height, width), np.uint8)
	cv2.polylines(mask, [left_eye_region], True, 255, 2)
	cv2.fillPoly(mask, [left_eye_region], 255)
	eye = cv2.bitwise_and(gray, gray, mask=mask)

	min_x = np.min(left_eye_region[:, 0])
	max_x = np.max(left_eye_region[:, 0])
	min_y = np.min(left_eye_region[:, 1])
	max_y = np.max(left_eye_region[:, 1])

	gray_eye = eye[min_y: max_y, min_x: max_x]
	_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
	height, width = threshold_eye.shape
	left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
	left_side_white = cv2.countNonZero(left_side_threshold)

	right_side_threshold = threshold_eye[0: height, int(width / 2): width]
	right_side_white = cv2.countNonZero(right_side_threshold)

	if left_side_white == 0:
		gaze_ratio = 1
	elif right_side_white == 0:
		gaze_ratio = 5
	else:
		gaze_ratio = left_side_white / right_side_white
	return gaze_ratio


def detect_closed_eyes(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    cv2.line(frame, left_point, right_point, (0, 255, 0), 1)
    cv2.line(frame, center_top, center_bottom, (0, 255, 0), 1)
    return ((math.hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))) / (math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))))

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def distance(a, b, c, d):
	return pow(pow((c - a), 2) + pow((d - b), 2), (1 / 2)) 

def some_function(gray, face):
	global blink_flag
	global blink_counter
	eye_counter=0
	global some_counter
	global another_counter
	global EYE_AR_CONSEC_FRAMES, EYE_AR_THRESH
	shape = predictor(gray, face)

	cv2.circle(frame, (shape.part(33).x, shape.part(33).y), 2, (0, 255, 255), 2) #tip of the nose
	cv2.circle(frame, (shape.part(2).x, shape.part(2).y), 2, (0, 255, 255), 2)
	cv2.circle(frame, (shape.part(14).x, shape.part(14).y), 2, (0, 255, 255), 2) 
	left_distance = distance(shape.part(33).x, shape.part(33).y, shape.part(2).x, shape.part(2).y)
	right_distance = distance(shape.part(33).x, shape.part(33).y, shape.part(14).x, shape.part(14).y)
	cv2.line(frame, (shape.part(33).x, shape.part(33).y), (shape.part(2).x, shape.part(2).y), (0, 255, 0), 1)
	cv2.line(frame, (shape.part(33).x, shape.part(33).y), (shape.part(14).x, shape.part(14).y), (0, 255, 0), 1)
	#print(left_distance, right_distance)
	if right_distance>=80:
		cv2.putText(frame, 'right', (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	elif left_distance>=80:
		cv2.putText(frame, 'left', (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
	left_eye_ratio = detect_closed_eyes([36, 37, 38, 39, 40, 41], shape)
	right_eye_ratio = detect_closed_eyes([42, 43, 44, 45, 46, 47], shape)
	blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
	if blinking_ratio > 5:
		cv2.putText(frame, "blink counter: "+ str(blink_counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		blink_counter+=1
		some_counter+=1
		cv2.putText(frame, "eyes closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, str(some_counter/fps_int), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	else:
		cv2.putText(frame, "eyes open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "blink frame "+ str(blink_counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		some_counter=0

	# Gaze detection
	gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], shape)
	gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], shape)
	gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2


	if blinking_ratio <3.9:
		
		if gaze_ratio <= 0.4:
			cv2.putText(frame, "RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
		elif 0.4 < gaze_ratio < 1.5:
			cv2.putText(frame, "CENTER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
		else:
			cv2.putText(frame, "LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
		


	shape = face_utils.shape_to_np(shape)
	(x, y, w, h) = face_utils.rect_to_bb(face)
	eye_grayscale = gray[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(eye_grayscale, scaleFactor=1.02, minNeighbors=20, minSize=(10,10))
	for (a,b,c,d) in eyes:
		eye_counter += 1
	if eye_counter ==1:
		cv2.putText(frame, str(eye_counter)+ " eye detected", (125,125),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	left_eye, right_eye, mouth = shape[lStart:lEnd], shape[rStart:rEnd], shape[mStart:mEnd]

	mouth_ear = mouth_aspect_ratio(mouth)
	cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [ cv2.convexHull(mouth)], -1, (0, 255, 0), 1)
	if mouth_ear > 1.75:
		cv2.putText(frame, "YAWNING ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.circle(frame, (x, y+h), 1, (0, 0, 255), 2)
	

	
	for (x, y) in shape:
		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

while True:
	ret, frame = cap.read()
	if ret == False:
		break

	face_counter =0
	area_face = []
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 1)
	for face in faces:
		face_counter +=1
		(x, y, w, h) = face_utils.rect_to_bb(face)
		area_face.append(w*h)
	if face_counter!=0:
		cv2.putText(frame, str(face_counter) + " Faces detected", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	else:
		cv2.putText(frame, "No Faces detected", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	for (i, face) in enumerate(faces):
		(x, y, w, h) = face_utils.rect_to_bb(face)
		if w*h == max(area_face):
			some_function(gray, face)
 
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("drowsiness.avi", fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)
	if writer is not None:
		writer.write(frame)	
	cv2.imshow("Output", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break
cv2.destroyAllWindows()
cap.release()

if writer is not None:
	writer.release()
