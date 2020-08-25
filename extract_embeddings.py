# USAGE
# python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
#	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
'''

class extract_embeddings:
	def __init__(self):

		# load our serialized face embedding model from disk
		print("[INFO] loading face recognizer...")
		self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

	def extract(self):
		# grab the paths to the input images in our dataset
		print("[INFO] quantifying faces...")
		self.imagePaths = list(paths.list_images("dataset"))

		# initialize our lists of extracted facial embeddings and
		# corresponding people names
		knownEmbeddings = []
		knownNames = []

		# initialize the total number of faces processed
		total = 0

		# loop over the image paths
		for (i, imagePath) in enumerate(self.imagePaths):
			# extract the person name from the image path
			print("[INFO] processing image {}/{}".format(i + 1,
				len(self.imagePaths)))
			name = imagePath.split(os.path.sep)[-2]

			# load the image, resize it to have a width of 600 pixels (while
			# maintaining the aspect ratio), and then grab the image
			# dimensions
			image = cv2.imread(imagePath)
			image = imutils.resize(image)#, width=600)
			(h, w) = image.shape[:2]

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			self.embedder.setInput(faceBlob)
			vec = self.embedder.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

		# dump the facial embeddings + names to disk
		#print("[INFO] serializing {} encodings...".format(total))
		data = {"embeddings": knownEmbeddings, "names": knownNames}
		f = open("output/embeddings.pickle", "wb")
		f.write(pickle.dumps(data))
		f.close()