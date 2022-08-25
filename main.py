from modnet import *
import glob
import os
import sys
import argparse
import cv2
import math
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

dir_path = os.path.dirname(os.path.realpath(__file__))

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser(description='InPainting')
parser.add_argument('--image', help='image path.')
args = parser.parse_args()

# Read and store arguments
confThreshold = 0.3
nmsThreshold = 0.9
inpWidth = 320
inpHeight = 320
model = "models/frozen_east_text_detection.pb"
text_detection_net = cv2.dnn.readNet(model)
text_detection_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")


############ Utility functions ############
def decode(scores, geometry, scoreThresh):
	detections = []
	confidences = []

	############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
	assert len(scores.shape) == 4, "Incorrect dimensions of scores"
	assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
	assert scores.shape[0] == 1, "Invalid dimensions of scores"
	assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
	assert scores.shape[1] == 1, "Invalid dimensions of scores"
	assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
	assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
	assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
	height = scores.shape[2]
	width = scores.shape[3]
	for y in range(0, height):

		# Extract data from scores
		scoresData = scores[0][0][y]
		x0_data = geometry[0][0][y]
		x1_data = geometry[0][1][y]
		x2_data = geometry[0][2][y]
		x3_data = geometry[0][3][y]
		anglesData = geometry[0][4][y]
		for x in range(0, width):
			score = scoresData[x]

			# If score is lower than threshold score, move to next x
			if(score < scoreThresh):
				continue

			# Calculate offset
			offsetX = x * 4.0
			offsetY = y * 4.0
			angle = anglesData[x]

			# Calculate cos and sin of angle
			cosA = math.cos(angle)
			sinA = math.sin(angle)
			h = x0_data[x] + x2_data[x]
			w = x1_data[x] + x3_data[x]

			# Calculate offset
			offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

			# Find points for rectangle
			p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
			p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
			center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
			detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
			confidences.append(float(score))

	# Return detections and confidences
	return [detections, confidences]


def inpaint():
	try:
		# register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log
		predict_config_path = os.path.join("configs", 'default.yaml')
		with open(predict_config_path, 'r') as f:
			predict_config = OmegaConf.create(yaml.safe_load(f))

		device = torch.device(predict_config.device)

		train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
		with open(train_config_path, 'r') as f:
			train_config = OmegaConf.create(yaml.safe_load(f))

		train_config.training_model.predict_only = True
		train_config.visualizer.kind = 'noop'

		out_ext = predict_config.get('out_ext', '.png')

		checkpoint_path = os.path.join(predict_config.model.path,
									   predict_config.model.checkpoint)
		model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
		model.freeze()
		if not predict_config.get('refine', False):
			model.to(device)

		if not predict_config.indir.endswith('/'):
			predict_config.indir += '/'

		dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
		for img_i in tqdm.trange(len(dataset)):
			mask_fname = dataset.mask_filenames[img_i]
			cur_out_fname = os.path.join(
				predict_config.outdir,
				os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
			)
			os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
			batch = default_collate([dataset[img_i]])
			if predict_config.get('refine', False):
				assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
				# image unpadding is taken care of in the refiner, so that output image
				# is same size as the input image
				cur_res = refine_predict(batch, model, **predict_config.refiner)
				cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
			else:
				with torch.no_grad():
					batch = move_to_device(batch, device)
					batch['mask'] = (batch['mask'] > 0) * 1
					batch = model(batch)
					cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
					unpad_to_size = batch.get('unpad_to_size', None)
					if unpad_to_size is not None:
						orig_height, orig_width = unpad_to_size
						cur_res = cur_res[:orig_height, :orig_width]

			cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
			cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
			cv2.imwrite(cur_out_fname, cur_res)

	except KeyboardInterrupt:
		return
	except Exception as ex:
		sys.exit(1)

if __name__ == "__main__":
	if args.image is not None:
		imagepath = args.image
	else:
		print("[Usage] main.py --image <image file path>")
		sys.exit()
	input_image = cv2.imread(imagepath)
	org_image = input_image.copy()
	input_image, mask_image = removeBackground(input_image)

	# Get frame height and width
	height_ = input_image.shape[0]
	width_ = input_image.shape[1]
	rW = width_ / float(inpWidth)
	rH = height_ / float(inpHeight)
	# Create a 4D blob from frame.
	blob = cv2.dnn.blobFromImage(input_image, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

	# Run the model
	text_detection_net.setInput(blob)
	output = text_detection_net.forward(outputLayers)
	# Get scores and geometry
	scores = output[0]
	geometry = output[1]
	[boxes, confidences] = decode(scores, geometry, confThreshold)
	# Apply NMS
	indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		# get 4 corners of the rotated rect
		vertices = cv2.boxPoints(boxes[i[0]])
		# scale the bounding box coordinates based on the respective ratios
		for j in range(4):
			vertices[j][0] *= rW
			vertices[j][1] *= rH
		points = []
		for j in range(4):
			p1 = (vertices[j][0], vertices[j][1])
			#p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
			points.append([vertices[j][0], vertices[j][1]])

		points = np.array([points], np.int32)
		cv2.fillConvexPoly(input_image, points, (255, 255, 255))
		cv2.fillConvexPoly(mask_image, points, 255)

	cv2.imwrite("test_inpaint_images/input.png", org_image)
	cv2.imwrite("test_inpaint_images/input_mask.png", mask_image)

	inpaint()
