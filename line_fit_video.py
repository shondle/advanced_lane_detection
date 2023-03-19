import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
from moviepy.editor import VideoFileClip
import pyrealsense2 as rs


# Global variables (just to make the moviepy video annotation work)
"""This would be good to adjust if we want to feed in realsense frames with a smaller width and height"""
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
window_size = 5  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # did the fast line fit detect the lines?
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None  # for calculating curvature



def annotate_image(img_in, depth_frame):
	"""
	Annotate the input image with lane line markings
	Returns annotated image
	"""
	global mtx, dist, left_line, right_line, detected
	global left_curve, right_curve, left_lane_inds, right_lane_inds

	# Undistort, threshold, perspective transform
	undist = cv2.undistort(img_in, mtx, dist, None, mtx)
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)	

	# Perform polynomial fit
	if not detected:
		# Slow line fit
		ret = line_fit(binary_warped)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Get moving average of line fit coefficients
		left_fit = left_line.add_fit(left_fit)
		right_fit = right_line.add_fit(right_fit)

		# Calculate curvature
		left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

		detected = True  # slow line fit always detects the line

	else:  # implies detected == True
		# # Fast line fit
		left_fit = left_line.get_fit()
		right_fit = right_line.get_fit()
		ret = tune_fit(binary_warped, left_fit, right_fit)
		try: 
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']
		except:
			print("failed")

		# Only make updates if we detected lines in current frame
		if ret is not None:
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

			left_fit = left_line.add_fit(left_fit)
			right_fit = right_line.add_fit(right_fit)
			left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
		else:
			detected = False

	# Calculate the offset of the vehicle from the middle.
	# vehicle_offset is in pixels, offset is in meters
	# (use offset for IGVC because it is in meters)

	vehicle_offset, offset, left_line_data, right_line_data = calc_vehicle_offset(undist, left_fit, right_fit, depth_frame)

	# Perform final visualization on top of original undistorted image
	result = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

	return result, offset, left_line_data, right_line_data


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':

	""" running the realsense camera and feeding every frame into the lane detection program"""
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # this was 640/280 earlier, but had to change (hurts computation though)
	config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

	first_frame = 1

	# Start streaming
	pipeline.start(config)


	try:
		while True:
			# Wait for a coherent pair of frames: depth and color
			frames = pipeline.wait_for_frames()
			depth_frame = frames.get_depth_frame()
			color_frame = frames.get_color_frame()
			if not depth_frame or not color_frame:
				continue

			# Convert images to numpy arrays
			depth_image = np.asanyarray(depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())


			# Put the CV lanes on the image, and return the offset of the vehicle from the middle. If it can't detect lanes, print error message.
			# it also returns all line data along the left line, and all line data along the right line. Do with this as you choose
			try:	
				binary_image, offset, left_line_data, right_line_data = annotate_image(color_image, depth_frame)
			except:
				print("Distortion amount exceeded")

			linedata = {}
			line_num = 1

			print(f"The offset value is {offset}")

			# Show the raw and annotated image
			cv2.imshow("Binary Image", binary_image)
			cv2.imshow("Line Detection", color_image)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q') or key == 27:
				break
	finally:
		pipeline.stop()
		cv2.destroyAllWindows()


	
	# Annotate a sample video instead of realsense camera input by running this example
	# annotate_video('project_video.mp4', 'out.mp4')

	# Annotate sample image instead of realsense camera input by running this example
	# Show example annotated image on screen for sanity check
	'''
	img_file = 'test_images/test4.jpg'
	img_file = 'test_images/test4.jpg'
	img = mpimg.imread(img_file)
	print(img.shape)
	result = annotate_image(img)
	result = annotate_image(img)
	result = annotate_image(img)
	plt.imshow(result)
	plt.show()
	'''
