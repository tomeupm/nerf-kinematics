import argparse
import os
import json
import sys
# Importar el módulo de preprocesamiento
from preprocessing import process_images_and_positions


def main():
    """Función principal del programa."""
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description='Herramienta para reconstrucción 3D usando NeRFs a partir de imágenes y datos de posición.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Agregar argumentos
    parser.add_argument('--img', metavar='PATH', type=str, required=True,
                        help='Ruta al directorio que contiene las imágenes')
    parser.add_argument('--pos', metavar='PATH', type=str, required=False, default=None,
                        help='Ruta al directorio que contiene los archivos JSON de posición (opcional)')
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Utilizar CUDA para acelerar el proceso con instant-ngp (si está disponible)')
    
    # Parsear los argumentos
    args = parser.parse_args()
    
    # Procesar las imágenes y posiciones usando el módulo de preprocesamiento
    processed_data = process_images_and_positions(args.img, args.pos, args.use_cuda)
    
    # TODO: Implementar la reconstrucción 3D con NeRFs usando los datos procesados
    print(f"Se han procesado {len(processed_data)} pares de imágenes y datos de posición.")
    
    # Seleccionar el método de reconstrucción según disponibilidad de CUDA
    if args.use_cuda:
        print("Utilizando instant-ngp para la reconstrucción 3D (aceleración con CUDA).")
        # TODO: Implementar la reconstrucción 3D con instant-ngp
    else:
        print("Utilizando NeRF puro para la reconstrucción 3D (CPU/GPU sin optimizaciones de instant-ngp).")
        # TODO: Implementar la reconstrucción 3D con NeRF puro
    
    print("Nota: La implementación completa de la reconstrucción 3D está pendiente.")


# Ejecutar el programa cuando se llame como script principal
if __name__ == '__main__':
    main()
