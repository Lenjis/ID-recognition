from PIL import Image
import os
import utils

# img = Image.new("L", (28, 28))

img = utils.preprocess_image("data\\images\\img_1111.jpg")
img.reshape(28, 28 * 14)
img = (img * 255).astype('uint8')
img_o = Image.fromarray(img)
img_o.save("output.png")
