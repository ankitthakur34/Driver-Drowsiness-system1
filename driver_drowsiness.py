from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

#26 inialize music file
mixer.init()
mixer.music.load("music.wav")

#6 calculate EAR of eyes
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
	#19 theshold value 0.25
thresh = 0.25
#22 frames after which we have to ring the alarm
frame_check = 20
#4 we use this because it is more eficient and fast then openCV harcascade
detect = dlib.get_frontal_face_detector()
#5 predict the 68 landmark of face, we need only eyes in this project so we will only calculate EAR 
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#12 landmark of left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

#1 capure live video frames and 0 means our primary camera
cap=cv2.VideoCapture(0)
#20 flag is frame count
flag=0
while True:
	#2 read gives us two values 1st bollean (ret) and 2nd gives us image frame
	ret, frame=cap.read()
	#24 resize the image/camera screen
	frame = imutils.resize(frame, width=850)
	#7 convert image frames in grey
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#8 decect grey frames and store them in subjects
	subjects = detect(gray, 0)
	#9 run a loop for every frame detect
	for subject in subjects:
		#10 predict the landmark of frames
		shape = predict(gray, subject)
		#11 converting shapes/landmark into x-y coordinate
		shape = face_utils.shape_to_np(shape)
		#13 detect left and right eye
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		#14 calculate EAR of left and right eye
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		#15 calaculate average of left and right eye EAR
		ear = (leftEAR + rightEAR) / 2.0
		#16 convexHull- gives minimal boundary to wrap the object 
		# wraps the left and right eye completly
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		#17 joins all the points has 5parameter 1st-image,2nd [points] in list,3rd -1 to draw,4th color,5th thickness
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		#18 condition if ear is less the threshold eye is closed ring the alarm
		if ear < thresh:
			#21 increses the frame count and print it
			flag += 1
			print (flag)
			#22 if ear is less then threshold value for long time then we have to give the warning and ring the alarm
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#play alarm 
				mixer.music.play()
				
				#23 if thes value is greater the set it to 0
		else:
			flag = 0
			#3 imshow displays image on windows take two parameter 1st window name and 2nd image
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	#25 close all window and destroy all things
cv2.destroyAllWindows()
cap.release() 