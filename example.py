import random
from Deep import Neural
import matplotlib.pyplot as plt
from PIL import Image
from timeit import default_timer as timer
from colorama import Fore, init
import numpy as np

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

def genSinapses(layers):
	brain = {
		'biases': [],
		'weights': []
	}

	brain['biases'] = [np.random.uniform(0, -1, x).reshape((1, x)) for x in layers[1:]]
	brain['weights'] = [np.random.uniform(-2, 2, x*y).reshape((x, y)) for x,y in zip(layers[:-1], layers[1:])]

	return brain

def main():
	compile_timer = timer()

	sinapses = genSinapses([28*28, 100, 15, 10])

	bia = Neural([28*28, 100, 15, 10], eta=0.1, gpu=True, brain=sinapses)
	bia2 = Neural([28*28, 100, 15, 10], eta=0.1, brain=sinapses)

	print(Fore.WHITE + 'compiler time =', timer() - compile_timer)
	processing_input = timer()

	images = read_image_files("data/train-images.idx3-ubyte")
	labels = read_label_files("data/train-labels.idx1-ubyte")

	print(Fore.WHITE + 'input processed: ', timer() - processing_input)
	total_time = timer()

	hit = 0
	hit2 = 0
	epoch_size = 100
	test = 0
	epoch = 0

	eixo_x = []
	eixo_y = []

	out = []
	for i in range(10):
		out.append(0.0)
	
	start = timer()

	acmb1 = 0.0
	acmb2 = 0.0

	for lazy in range(30):
		train_test = [(x,y) for x, y in zip(images,labels)]
		random.shuffle(train_test)
		images = [x for (x,y) in train_test]
		labels = [y for (x,y) in train_test]
		for i in range(60000):

			ans1 = bia.send(images[i])
			if isOk(labels[i], ans1):
				hit = hit + 1
			
			ans2 = bia2.send(images[i])
			if isOk(labels[i], ans2):
				hit2 = hit2 + 1

			test = test + 1
			out[labels[i]] = 1.0

			bia1_timer = timer()
			bia.learn(images[i], out)
			bia1_timer = timer() - bia1_timer
			
			bia2_timer = timer()
			bia2.learn(images[i], out)
			bia2_timer = timer() - bia2_timer

			bia1_timer *= 1000
			bia2_timer *= 1000

			acmb1 += bia1_timer
			acmb2 += bia2_timer

			acmb1 = round(acmb1, 3)
			acmb2 = round(acmb2, 3)

			out[labels[i]] = 0.0

			if test == epoch_size:
				rate = hit/epoch_size
				print(Fore.WHITE + "(bia1) epoch {}: rate = {}, learn = {}ms".format(epoch, rate, acmb1))
				rate = hit2/epoch_size
				print(Fore.WHITE + "(bia2) epoch {}: rate = {}, learn = {}ms".format(epoch, rate, acmb2))
				print(Fore.GREEN + "-------")
				acmb1 = 0.0
				acmb2 = 0.0
				start = timer()
				epoch = epoch + 1
				test = 0
				hit = 0
				hit2 = 0
				eixo_x.append(epoch)
				eixo_y.append(rate)

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

	print(Fore.WHITE + "Total score: {}/{} => {}".format(hit,10000,hit/10000))

	pick = random.randint(0,9999)
	print_image = images[pick]
	ans = bia.send(print_image)
	ans = organize(ans)
	print(Fore.WHITE + "Esperado: {}".format(labels[pick]))
	print(Fore.WHITE + "Chute: {} com {}%% de precisao\n {} com {}%% de precisao.\n {} com {}%% de precisao".format(ans[0][0],(ans[0][1]*100)//1,ans[1][0],(ans[1][1]*100.0)//1,ans[2][0],(ans[2][1]*100.0)//1))
	print(Fore.WHITE + "total time: ",  timer() - total_time)
	teste(print_image)
	graph(eixo_x,eixo_y)

def teste(l):
	width = 28
	height = 28

	img = Image.new(mode="L",size=(width,height))
	for i in range(28):
		for j in range(28):
			img.putpixel((j,i),l[i*28 + j])
	img.show()

def graph(eixo_x,eixo_y):
	plt.plot(eixo_x,eixo_y,color = 'blue')
	plt.title('Cost vs test')
	plt.show()

if __name__ == "__main__":
	init()
	main()