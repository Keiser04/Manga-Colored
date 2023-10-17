import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from colorizator import MangaColorizator

def process_image(image, colorizator, args):
    colorizator.set_image(image, args.size, args.denoiser, args.denoiser_sigma)
    return colorizator.colorize()

def colorize_single_image(image_path, save_path, colorizator, args):
    image = plt.imread(image_path)
    colorization = process_image(image, colorizator, args)
    plt.imsave(save_path, colorization)

def colorize_images_in_folder(folder_path, colorizator, args):
    images = os.listdir(folder_path)
    output_folder = os.path.join(folder_path, 'colorization')
    os.makedirs(output_folder, exist_ok=True)
    
    for image_name in images:
        file_path = os.path.join(folder_path, image_name)
        
        if os.path.isdir(file_path) or not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        save_path = os.path.join(output_folder, image_name)
        colorize_single_image(file_path, save_path, colorizator, args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-gen", "--generator", default='/content/Manga-Colored/networks/generator.zip')
    parser.add_argument("-ext", "--extractor", default='networks/extractor.pth')
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true')
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false')
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=25)
    parser.add_argument("-s", "--size", type=int, default=576)
    parser.set_defaults(gpu=False)
    parser.set_defaults(denoiser=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        
    colorizer = MangaColorizator(device, args.generator, args.extractor)
    
    if os.path.isdir(args.path):
        colorize_images_in_folder(args.path, colorizer, args)
    else:
        print("La ruta especificada no es un directorio válido.")
