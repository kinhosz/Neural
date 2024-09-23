def calculate_pooling_layers(width, height):
    layers = 0

    while width > 1 or height > 1:
        layers += 1
        width = (width + 1) // 2
        height = (height + 1) // 2

    return layers
