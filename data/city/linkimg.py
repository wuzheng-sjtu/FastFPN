# script to soft link image in Cityscapes dataset
import os, sys

phase = ['train', 'val', 'test']
root_dir = '/ais/gobi3/datasets/cityscapes_release/leftImg8bit'
link_dir = '/ais/gobi5/videodet/FPN/data/city/'
for pha in phase:
    phase_dir = os.path.join(root_dir, pha)
    citys = os.listdir(phase_dir)
    for city in citys:
	city_dir = os.path.join(phase_dir, city)
	imgs = os.listdir(city_dir)
	for img in imgs:
	    source_path = os.path.join(city_dir, img)
	    target_path = os.path.join(link_dir, pha, img)
	    os.system('ln -s {:s} {:s}'.format(source_path, target_path))
    

