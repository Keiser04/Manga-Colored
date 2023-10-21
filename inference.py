import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from colorizator import MangaColorizator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("--eliminar", dest='eliminar', action='store_true', help='Eliminar imágenes en blanco y negro después de procesarlas')
    parser.add_argument("-s", "--size", type=int, default=1088)
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false')
    parser.add_argument("-o", "--output", help='Ruta de salida para las imágenes coloreadas')
    args = parser.parse_args()
    return args

def colorize_single_image(image_path, output_path, colorizer, args):
    image = plt.imread(image_path)
    colorizer.set_image(image, args.size, not args.denoiser, args.size)  # No aplicamos denoise si args.denoiser es False
    colorization = colorizer.colorize()
    plt.imsave(output_path, colorization)

    # Eliminar la imagen en blanco y negro después de procesarla si la opción 'eliminar' es True
    if args.eliminar:
        os.remove(image_path)

if __name__ == "__main__":
    args = parse_args()

    colorizer = MangaColorizator('cuda')  # Suponiendo que necesitas usar la GPU

    if os.path.isdir(args.path):
        if args.output is None:
            output_dir = args.path  # Usar la misma ubicación que las imágenes de entrada
        else:
            output_dir = args.output

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images = os.listdir(args.path)

        for image_name in images:
            file_path = os.path.join(args.path, image_name)

            if os.path.isdir(file_path):
                continue

            name, ext = os.path.splitext(image_name)
            if ext.lower() not in ['.png', '.jpg', '.jpeg']:
                continue  # Saltar archivos que no son imágenes

            save_path = os.path.join(output_dir, image_name)
            colorize_single_image(file_path, save_path, colorizer, args)

        print("Procesamiento y eliminación de imágenes completados.")
    else:
        print('Ruta incorrecta o no es un directorio')
