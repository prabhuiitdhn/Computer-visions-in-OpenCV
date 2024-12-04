from PIL import Image
from PIL import ImageFilter, ImageEnhance

kelvin_table = {
    1000: (255, 56, 0),
    1500: (255, 109, 0),
    2000: (255, 137, 18),
    2500: (255, 161, 72),
    3000: (255, 180, 107),
    3500: (255, 196, 137),
    4000: (255, 209, 163),
    4500: (255, 219, 186),
    5000: (255, 228, 206),
    5500: (255, 236, 224),
    6000: (255, 243, 239),
    6500: (255, 249, 253),
    7000: (245, 243, 255),
    7500: (235, 238, 255),
    8000: (227, 233, 255),
    8500: (220, 229, 255),
    9000: (214, 225, 255),
    9500: (208, 222, 255),
    10000: (204, 219, 255)}


def convert_temp(image, temp):
    r, g, b = kelvin_table[temp]
    matrix = (r / 255.0, 0.0, 0.0, 0.0,
              0.0, g / 255.0, 0.0, 0.0,
              0.0, 0.0, b / 255.0, 0.0)
    return image.convert('RGB', matrix)


def reduced_color(image):
    color = ImageEnhance.Color(image)
    return color.enhance(0.75)


def reduced_contrast(image):
    color = ImageEnhance.Contrast(image)
    return color.enhance(0.25)


def reduced_brightness(image):
    brightness = ImageEnhance.Brightness(image)
    return brightness.enhance(0.75)


input_image_path = r"D:\Camera-Blockage\mud-splash\Day-to-night\image_000001.jpg"
image = Image.open(input_image_path)


image = reduced_color(image)
image.save("./reduced_color_image.jpg")

# image = reduced_contrast(image)
# image.save("./reduced_contrast_image.jpg")

image = convert_temp(image, 2500)
image.save("./output_image.jpg")

image = reduced_brightness(image)
image.save("./reduced_brightness.jpg")