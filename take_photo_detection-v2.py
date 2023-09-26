# import the necessary packages
import argparse

import cv2
import imutils
import pytesseract
from imutils.video import FPS
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\PC\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'


def ProcessImage(frame, test=0):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	noise = cv2.medianBlur(gray, 3)
	# thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	# show the output frame
	# cv2.imshow("Photo Text Detection [PROCESSED] " + test.__str__(), noise)

	results = pytesseract.image_to_data(noise, output_type=Output.DICT, lang='por', config='--psm 3 --oem 3')

	parsedResults = []
	# loop over each of the individual text localizations
	for i in range(0, len(results["text"])):
		# extract the bounding box coordinates of the text region from
		# the current result
		x = results["left"][i]
		y = results["top"][i]
		w = results["width"][i]
		h = results["height"][i]
		# extract the OCR text itself along with the confidence of the
		# text localization
		text = results["text"][i].strip()
		conf = int(results["conf"][i])

		# filter out weak confidence text localizations
		if conf > 45 and len(text) > 0:
			# display the confidence and text to our terminal
			print("Confidence: {}".format(conf))
			print("Text: {}".format(text))
			print("")
			# strip out non-ASCII text so we can draw the text on the image
			# using OpenCV, then draw a bounding box around the text along
			# with the text itself
			parsedResults.append(
				{
					'text': text,
					'conf': conf,
					'x': x,
					'y': y,
					'order': i
				}
			)

			text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	# print(text)

	print(parsedResults)
	cv2.imshow("Photo Text Detection " + test.__str__(), frame)


if __name__ == '__main__':
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
					help="minimum probability required to inspect a region")
	ap.add_argument("-w", "--width", type=int, default=1000,
					help="resized image width (should be multiple of 32)")
	ap.add_argument("-e", "--height", type=int, default=1000,
					help="resized image height (should be multiple of 32)")
	args = vars(ap.parse_args())

	# initialize the original frame dimensions, new frame dimensions,
	# and ratio between the dimensions
	(newW, newH) = (args["width"], args["height"])

	vs = cv2.VideoCapture(1)
	fps = FPS().start()
	test = 0
	# loop over frames from the video stream
	while True:
		# grab the current frame, then handle if we are using a
		# VideoStream or VideoCapture object
		frame = vs.read()

		# check to see if we have reached the end of the stream
		if frame is None:
			break
		# resize the frame, maintaining the aspect ratio
		frame = imutils.resize(frame[1], width=1000)
		orig = frame.copy()

		cv2.imshow("VIDEO_CAMERA", orig)

		key = cv2.waitKey(1) & 0xFF
		if key == ord(" "):
			# resize the frame, this time ignoring aspect ratio
			frame = cv2.resize(frame, (newW, newH))

			ProcessImage(frame, test)

			test += 1

		if key == ord("q"):
			break

		fps.update()
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	vs.release()

	# close all windows
	cv2.destroyAllWindows()
