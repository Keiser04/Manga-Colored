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
    args = parser.parse_args()
    return args

def colorize_images_in_directory(directory, colorizer, args):
    for foldername, _, filenames in os.walk(directory):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext.lower() not in ['.png', '.jpg', '.jpeg']:
                continue  # Saltar archivos que no son imágenes

            file_path = os.path.join(foldername, filename)
            colorize_single_image(file_path, colorizer, args)

def colorize_single_image(image_path, colorizer, args):
    image = plt.imread(image_path)
    colorizer.set_image(image, args.size, not args.denoiser, args.size)  # No aplicamos denoise si args.denoiser es False
    colorization = colorizer.colorize()
    plt.imsave(image_path, colorization)  # Guardar la imagen coloreada en la misma ubicación que la original

    # Eliminar la imagen en blanco y negro después de procesarla si la opción 'eliminar' es True
    if args.eliminar:
        os.remove(image_path)

    print(f"Imagen coloreada guardada en: {image_path}")

if __name__ == "__main__":
    args = parse_args()

    colorizer = MangaColorizator('cuda')  # Suponiendo que necesitas usar la GPU

    if os.path.isdir(args.path):
        colorize_images_in_directory(args.path, colorizer, args)
        print("Procesamiento y eliminación de imágenes completados.")
    else:
        print('Ruta incorrecta o no es un directorio')
