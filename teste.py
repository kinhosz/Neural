import numpy as np
import random
import math
from model import Network
from lib import Function as F
from PIL import Image

def read_image_files(url):
	f = open(url,"rb")
	magic_number = int.from_bytes(f.read(4), byteorder="big")
	number_of_images = int.from_bytes(f.read(4), byteorder="big")
	rows = int.from_bytes(f.read(4),byteorder="big")
	columns = int.from_bytes(f.read(4),byteorder="big")

	images = []

	for img in range(number_of_images):
		image = []
		for i in range(rows):
			for j in range(columns):
				pixel = int.from_bytes(f.read(1),byteorder="big",signed=False)
				image.append(pixel)
		images.append(image)

	f.close()

	return images

def read_label_files(url):
	f = open(url,"rb")
	magic_number = int.from_bytes(f.read(4), byteorder="big")
	number_of_items = int.from_bytes(f.read(4), byteorder="big")

	labels = []

	for lbl in range(number_of_items):
		label = int.from_bytes(f.read(1),byteorder="big",signed=False)
		labels.append(label)

	f.close()

	return labels

def isOk(predict,l):
	
	pred = -1
	ans = -0.1

	for i in range(len(l)):
		if l[i] > ans:
			ans = l[i]
			pred = i

	return pred == predict

def organize(l):

	ret = []

	for i in range(3):
		best = -0.1
		actual = -1

		for j in range(10):
			if l[j] > best:
				best = l[j]
				actual = j

		ret.append((actual,best))
		l[actual] = -0.3

	return ret

def main():

	bia = Network([28*28,15,10],eta = 0.1)

	images = read_image_files("data/train-images.idx3-ubyte")
	labels = read_label_files("data/train-labels.idx1-ubyte")

	train_test = [(x,y) for x, y in zip(images,labels)]
	random.shuffle(train_test)
	images = [x for (x,y) in train_test]
	labels = [y for (x,y) in train_test]

	hit = 0
	epoch_size = 1000
	test = 0
	epoch = 0

	out = []
	for i in range(10):
		out.append(0.0)

	for i in range(60000):

		ans = bia.send(images[i])
		if isOk(labels[i],ans):
			hit = hit + 1

		test = test + 1
		out[labels[i]] = 1.0
		bia.learn(images[i],out)
		out[labels[i]] = 0.0

		if test == epoch_size:
			rate = hit/epoch_size
			print("epoch {}: rate = {}".format(epoch,rate))
			epoch = epoch + 1
			test = 0
			hit = 0

	images = read_image_files("data/t10k-images.idx3-ubyte")
	labels = read_label_files("data/t10k-labels.idx1-ubyte")

	train_test = [(x,y) for x, y in zip(images,labels)]
	random.shuffle(train_test)
	images = [x for (x,y) in train_test]
	labels = [y for (x,y) in train_test]

	hit = 0
	epoch_size = 1000
	test = 0
	epoch = 0

	out = []
	for i in range(10):
		out.append(0.0)

	for i in range(10000):

		ans = bia.send(images[i])
		acerto = False
		if isOk(labels[i],ans):
			hit = hit + 1
			acerto = True

		org = organize(ans)

		if acerto:
			print("[OK] => {}".format(org))
		else:
			print("[ERROR - {}] => {}".format(labels[i],org))

	print("Total score: {}/{} => {}".format(hit,10000,hit/10000))

	pick = random.randint(0,9999)
	print_image = images[pick]
	ans = bia.send(print_image)
	ans = organize(ans)
	print("Esperado: {}".format(labels[pick]))
	print("Chute: {} com {}%% de precisao\n {} com {}%% de precisao. {} com {}%% de precisao".format(ans[0][0],(ans[0][1]*100)//1,ans[1][0],(ans[1][1]*100.0)//1,ans[2][0],(ans[2][1]*100.0)//1))
	teste(print_image)

def teste(l):
	width = 28
	height = 28

	img = Image.new(mode="L",size=(width,height))
	for i in range(28):
		for j in range(28):
			img.putpixel((j,i),l[i*28 + j])
	img.show()

if __name__ == "__main__":
	main()