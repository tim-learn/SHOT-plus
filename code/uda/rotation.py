import torch
import torch.utils.data
from torchvision import datasets
import numpy as np

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)


def rotate_single_with_label2(img, label):
	if label == 1:
		img = tensor_rot_180(img)
	return img

def rotate_batch_with_labels2(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label2(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)


def rotate_batch(batch, label='rand'):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels

class RotateImageFolder(datasets.ImageFolder):
	def __init__(self, traindir, train_transform, original=True, rotation=True, rotation_transform=None):
		super(RotateImageFolder, self).__init__(traindir, train_transform)
		self.original = original
		self.rotation = rotation
		self.rotation_transform = rotation_transform		

	def __getitem__(self, index):
		path, target = self.imgs[index]
		img_input = self.loader(path)

		if self.transform is not None:
			img = self.transform(img_input)
		else:
			img = img_input

		results = []
		if self.original:
			results.append(img)
			results.append(target)
		if self.rotation:
			if self.rotation_transform is not None:
				img = self.rotation_transform(img_input)
			target_ssh = np.random.randint(0, 4, 1)[0]
			img_ssh = rotate_single_with_label(img, target_ssh)
			results.append(img_ssh)
			results.append(target_ssh)
		return results

	def switch_mode(self, original, rotation):
		self.original = original
		self.rotation = rotation
