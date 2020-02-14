import argparse
import os
import models
from util import util
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/home/iceclear/Video-compression/SVEGAN/results/test_gt')
parser.add_argument('-d1','--dir1', type=str, default='/home/iceclear/Video-compression/SVEGAN/results/test_32_nb3')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# crawl directories
# f = open(opt.out,'w')
files = os.listdir(opt.dir0)
score_list = []
video_list = {}
ave_dic = {}
for file in files:
	name_array = file.split('_')
	keyname = name_array[0]+name_array[1]
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1,file)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = model.forward(img0,img1)

		if keyname not in video_list.keys():
			video_list[keyname] = []

		video_list[keyname].append(dist01.detach().numpy())
		score_list.append(dist01.detach().numpy())
		# print('%s: %.3f'%(file,dist01))
		# f.writelines('%s: %.6f\n'%(file,dist01))

for key_name in video_list.keys():
	ave_dic[key_name]=np.mean(np.array(video_list[key_name]))

print('\n*****************************************\n')
print(opt.dir1)
print(ave_dic)
score_list = np.array(score_list)
average_score = np.mean(score_list)
print('\nFinal average score:%.3f'%(average_score))
print('\n*****************************************\n')

# f.close()
