import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from data import get_celebA, flags
from model import get_generator, get_discriminator
import helper
import matplotlib.pyplot as plt
plt.switch_backend('agg')

if __name__ == '__main__':
	G = get_generator([None, flags.z_dim])
	G.load_weights('checkpoint/G12.npz')
	plt.figure(figsize=(6, 6), dpi=100)
	for i in range(500):
		z = np.random.normal(loc=0.0, scale=1.0, size=[9, flags.z_dim]).astype(np.float32)
		G.eval()
		result = G(z)
		G.train()
		plt.imshow(helper.images_square_grid(result.numpy()))
		plt.axis("off")
		imagePath = '{}/{:003d}_image.png'.format(flags.sample_dir, i+1)
		if os.path.exists(imagePath):
			continue
		plt.savefig(imagePath, bbox_inches='tight', pad_inches=0)
