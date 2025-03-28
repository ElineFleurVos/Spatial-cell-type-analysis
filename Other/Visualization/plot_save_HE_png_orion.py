import matplotlib.pyplot as plt
from PIL import Image
import os 
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms
import skimage.exposure
import numpy as np 

#RUN in tiatoolbox environment

def preproc_func_gamma(img):
    img_np = np.array(img)
    img_gamma = skimage.exposure.adjust_gamma(img_np, 2.2)
    img_tensor = torchvision.transforms.ToTensor()(img_gamma).permute(1, 2, 0)
    img_pil = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8))
    return img_pil

output_dir = "/home/evos/Outputs/CRC/Orion/Orion_HE_png"
output_dir_gamma = "/home/evos/Outputs/CRC/Orion/Orion_HE_png_gamma"\

slides_dir = "/home/evos/Data/CRC/Orion_slides"
slides = os.listdir(slides_dir)
counter = 1
for filename in os.listdir(slides_dir):
    print(counter)
    slide_dir = os.path.join(slides_dir, filename)
    file_name = slide_dir.split("/")[-1]
    slide_name = file_name.split(".")[0]
    
    image = Image.open(slide_dir)
    new_width = int(image.width * 0.01)
    new_height = int(image.height * 0.01)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # plt.imshow(resized_image)
    # plt.axis('off')  # Hide axis
    # plt.show()  

    resized_image_gamma = preproc_func_gamma(resized_image)
    
    # plt.imshow(resized_image_gamma)
    # plt.axis('off')  # Hide axis
    # plt.show()

    resized_image.save(f"{output_dir}/{slide_name}.png")
    resized_image_gamma.save(f"{output_dir_gamma}/{slide_name}.png")   
    counter += 1 

print("Done saving png files")     