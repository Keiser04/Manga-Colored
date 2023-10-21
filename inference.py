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

        # Eliminar la imagen en blanco y negro después de procesarla si la opción 'eliminar' es True
        if args.eliminar:
            os.remove(file_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("--eliminar", dest='eliminar', action='store_true', help='Eliminar imágenes en blanco y negro después de procesarlas')
    parser.add_argument("-s", "--size", type=int, default=1088)
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false')
    parser.add_argument("-o", "--output", required=True, help='Ruta de salida para las imágenes coloreadas')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    colorizer = MangaColorizator('cuda')  # Suponiendo que necesitas usar la GPU
    
    if os.path.isdir(args.path):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
              
        colorize_images(args.output, colorizer, args)
    else:
        print('Ruta incorrecta o no es un directorio')
