import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

#
import PIL.Image as pil_image
from math import log10
import pytorch_ssim

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
#
parser.add_argument('--image_adress', default='/content/drive/MyDrive/SYS843/Github/sr/datasets/Set5/', type=str, help='image adress')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

#
IMAGE_ADRESS = opt.image_adress

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_ADRESS + IMAGE_NAME)

#
hr_width = (image.width // UPSCALE_FACTOR) * UPSCALE_FACTOR
hr_height = (image.height // UPSCALE_FACTOR) * UPSCALE_FACTOR
hr = image.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
lr = image.resize((hr_width // UPSCALE_FACTOR, hr_height // UPSCALE_FACTOR), resample=pil_image.BICUBIC)

image_lr = Variable(ToTensor()(lr), volatile=True).unsqueeze(0)
image_hr = Variable(ToTensor()(hr), volatile=True).unsqueeze(0)
if TEST_MODE:
    image_lr = image_lr.cuda()
    image_hr = image_hr.cuda()


start = time.clock()
image_sr = model(image_lr)
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(image_sr[0].data.cpu())

#print result

mse = ((image_hr - image_sr) ** 2).data.mean()
psnr = 10 * log10(1 / mse)
#ssim = pytorch_ssim.ssim(image_sr, image_hr).data[0]

print('psnr' + str(psnr))

#save
out_adress = '/content/drive/MyDrive/SYS843/Github/sr/SRGAN_leftt/data/out_'+ str(UPSCALE_FACTOR) + '/'
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
hr.save('out_truth_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
