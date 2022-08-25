import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

sys.path.append("./BackgroundReplace/")
from src.models.modnet import MODNet
torch_transforms = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	]
)

dir_path = os.path.dirname(os.path.realpath(__file__))

print('Load pre-trained MODNet...')
pretrained_ckpt = dir_path + '/BackgroundReplace/model/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
	print('Use GPU...')
	modnet = modnet.cuda()
	modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
	print('Use CPU...')
	modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()

def replaceBackground(frame_np, orgw, orgh, background_image):
	frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
	org = frame_np.copy()
	frame_np = cv2.resize(frame_np, (512, 512), cv2.INTER_AREA)

	frame_PIL = Image.fromarray(frame_np)
	frame_tensor = torch_transforms(frame_PIL)
	frame_tensor = frame_tensor[None, :, :, :]
	if GPU:
		frame_tensor = frame_tensor.cuda()

	with torch.no_grad():
		_, _, matte_tensor = modnet(frame_tensor, True)

	matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
	matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
	matte_np = cv2.resize(matte_np, (orgw, orgh))
	fg_np = (matte_np * org).astype(np.uint8)
	bg_frame = ((1 - matte_np) * background_image).astype(np.uint8)
	frame1 = fg_np + bg_frame
	return frame1

def removeBackground(frame_np):
	frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
	org = frame_np.copy()
	orgh, orgw,_ = org.shape
	frame_np = cv2.resize(frame_np, (512, 512), cv2.INTER_AREA)

	frame_PIL = Image.fromarray(frame_np)
	frame_tensor = torch_transforms(frame_PIL)
	frame_tensor = frame_tensor[None, :, :, :]
	if GPU:
		frame_tensor = frame_tensor.cuda()

	with torch.no_grad():
		_, _, matte_tensor = modnet(frame_tensor, True)

	matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
	matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
	matte_np = cv2.resize(matte_np, (orgw, orgh))
	matte_np1 = 1 - matte_np
	fg_np = (matte_np1 * org).astype(np.uint8)
	fg_np = cv2.cvtColor(fg_np, cv2.COLOR_RGB2BGR)
	matte_np = cv2.cvtColor(matte_np, cv2.COLOR_BGR2GRAY)
	return fg_np, matte_np * 255