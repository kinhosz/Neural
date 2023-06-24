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