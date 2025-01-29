import os
import random
from torchvision.io import read_image
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg') 

def random_folder_shuffle(folder="train"):
    """
    Randomly shuffles the images and masks in the given folder

    Params:
        folder: str - the folder to shuffle (choices: "train", "val", "test")
    Outputs:
        pairs: list - list of tuples containing the image and mask pairs
    """
    if folder not in ["train", "val", "test"]:
        print("Invalid folder name.")
        return
    elif folder == "train":
        directory = "data/train/"
    elif folder == "val":
        directory = "data/val/"
    else:
        directory = "data/test/"
    images = sorted([f for f in os.listdir(directory) if f.endswith(".jpg")])
    masks = sorted([f for f in os.listdir(directory) if f.endswith("_mask.png")])
    if len(images) < 5 or len(masks) < 5:
        print("Not enough images or masks in the directory.")
        return
    image_pairs=[]
    for i in range(len(images)):
        pair = (os.path.join(directory, images[i]),
                os.path.join(directory, masks[i]))
        image_pairs.append(pair)
    random.shuffle(image_pairs)
    return image_pairs

def display_image(pair,fig):
    plt.clf()
    image = read_image(pair[0])
    mask = read_image(pair[1])
    
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0))

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask.permute(1, 2, 0))
    
    plt.draw()
    plt.pause(0.001)

def loop_display_images(image_pairs):
    if not image_pairs:
        return
        
    fig = plt.figure(figsize=(12, 6))
    index = 0
    
    def on_key(event):
        nonlocal index
        if event.key == 'escape':
            plt.close()
        elif event.key == ' ':  # space key
            index += 1
            if index >= len(image_pairs):
                plt.close()
            else:
                display_image(image_pairs[index], fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    display_image(image_pairs[index], fig)
    plt.show()

if __name__ == "__main__":
    image_pairs=random_folder_shuffle("train")
    loop_display_images(image_pairs)