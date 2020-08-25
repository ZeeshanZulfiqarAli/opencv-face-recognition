# USAGE
# python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())'''

class recognize:
	def __init__(self,faceConf,recogConf):

		#confidence for face detection
		self.faceConf = faceConf
		#confidence for face recognition
		self.recogConf = recogConf
		self.previousDetection = None
		# load our serialized face detector from disk
		print("[INFO] loading face detector...")
		protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
		modelPath = os.path.sep.join(["face_detection_model",
			"res10_300x300_ssd_iter_140000.caffemodel"])
		self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		# load our serialized face embedding model from disk
		print("[INFO] loading face recognizer...")
		self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

		# load the actual face recognition model along with the label encoder
		self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
		self.le = pickle.loads(open("output/le.pickle", "rb").read())
		#to prevent race conditions
		self.pause = False

	def detect(self,image):
		image = imutils.resize(image)# width=600)
		(h, w) = image.shape[:2]
		
		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		self.detector.setInput(imageBlob)
		detections = self.detector.forward()
		startX = startY = endX = endY = proba = 0
		name = None
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > self.faceConf:

				# compute the (x, y)-coordinates of the bounding box for the
				# face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(tmpStartX, tmpStartY, tmpEndX, tmpEndY) = box.astype("int")

				# extract the face ROI
				face = image[tmpStartY:tmpEndY, tmpStartX:tmpEndX]
				#cv2.imshow("showing",face)
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue
				if self.pause:
					return image,(tmpStartX, tmpStartY, tmpEndX, tmpEndY),None,0
				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
					(0, 0, 0), swapRB=True, crop=False)
				self.embedder.setInput(faceBlob)
				vec = self.embedder.forward()

				# perform classification to recognize the face
				preds = self.recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				tmpProba = preds[j]
				tmpName = self.le.classes_[j]

				if tmpProba>proba:
					proba = tmpProba
					name = tmpName
					(startX, startY, endX, endY) = (tmpStartX,tmpStartY,tmpEndX,tmpEndY)

				# draw the bounding box of the face along with the associated
				# probability
				'''
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				
				cv2.putText(image, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)'''

		# show the output image
		if proba < self.recogConf:
			name = "~unknown"

		return image,(startX, startY, endX, endY),name,proba
		#cv2.imshow("Image", image)
		#cv2.waitKey(0)
	
	def namePresent(self,name):
		return name in self.le.classes_[:]
	
	def updateModel(self):
		self.pause = True
		# Reload the actual face recognition model along with the label encoder
		self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
		self.le = pickle.loads(open("output/le.pickle", "rb").read())
		self.pause = False