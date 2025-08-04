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
				image.append(pixel / 255)
		images.append(image)

	f.close()

	return images
