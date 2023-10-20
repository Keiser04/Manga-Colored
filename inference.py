import os
import argparse
import sys

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
    return True

def colorize_images(target_path, colorizator, args):
    images = os.listdir(args.path)
    
    for image_name in images:
        file_path = os.path.join(args.path, image_name)
        
        if os.path.isdir(file_path):
            continue
        
        name, ext = os.path.splitext(image_name)
        if (ext != '.png'):
            image_name = name + '.png'
        
        save_path = os.path.join(target_path, image_name)
        colorize_single_image(file_path, save_path, colorizator, args)

        # Eliminar la imagen en blanco y negro después de procesarla si el booleano 'eliminar' es True
        if args.eliminar:
            os.remove(file_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-o", "--output", default='colorized_images')
    parser.add_argument("--eliminar", dest='eliminar', action='store_true', help='Eliminar imágenes en blanco y negro después de procesarlas')
    parser.add_argument("-gen", "--generator", default='/content/Manga-Colored/networks/generator.zip')
    parser.add_argument("-ext", "--extractor", default='networks/extractor.pth')
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true')
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false')
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=10)
    parser.add_argument("-s", "--size", type=int, default=512)
    parser.set_defaults(gpu=True)
    parser.set_defaults(denoiser=True)
    parser.set_defaults(eliminar=False)  # Valor predeterminado para la opción eliminar
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
        colorization_path = os.path.join(args.path, args.output)
        if not os.path.exists(colorization_path):
            os.makedirs(colorization_path)
              
        colorize_images(colorization_path, colorizer, args)
        
    elif os.path.isfile(args.path):
        split = os.path.splitext(args.path)
        
        if split[1].lower() in ('.jpg', '.png', '.jpeg'):
            new_image_path = os.path.join(args.output, split[0] + '_colorized.png')
            colorize_single_image(args.path, new_image_path, colorizer, args)
            
            # Eliminar la imagen en blanco y negro después de procesarla si el booleano 'eliminar' es True
            if args.eliminar:
                os.remove(args.path)
        else:
            print('Wrong format')
    else:
        print('Wrong path')
