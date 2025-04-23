import os
import sys
import json
import shutil


def estimate_camera_positions(img_path, output_path=None):
    """
    Estima las posiciones de cámara usando COLMAP/GLOMAP.
    
    Args:
        img_path: Ruta al directorio que contiene las imágenes
        output_path: Ruta donde se guardarán los archivos JSON de posiciones estimadas
                    (si es None, se usa una carpeta 'estimated_positions' dentro de img_path)
    
    Returns:
        Ruta al directorio con las posiciones estimadas
    """
    # TODO: Implementar la lógica para llamar a COLMAP/GLOMAP
    print("Estimando posiciones de cámara con COLMAP/GLOMAP...")
    
    # Si no se especifica una ruta de salida, crear una dentro del directorio de imágenes
    if output_path is None:
        output_path = os.path.join(os.path.dirname(img_path), "estimated_positions")
        os.makedirs(output_path, exist_ok=True)
    
    # TODO: Ejecutar COLMAP/GLOMAP para estimar posiciones
    # Ejemplo pseudocódigo:
    # run_colmap(
    #     img_path=img_path, 
    #     output_path=output_path,
    #     features="sift",
    #     matcher="exhaustive"
    # )
    
    # TODO: Convertir la salida de COLMAP/GLOMAP a formato JSON
    # para cada imagen en img_path
    
    print(f"Posiciones de cámara estimadas y guardadas en: {output_path}")
    return output_path


def process_images_and_positions(img_path, pos_path=None, use_cuda=False):
    """
    Procesa las imágenes y datos de posición para la reconstrucción 3D con NeRFs.
    
    Args:
        img_path: Ruta al directorio que contiene las imágenes
        pos_path: Ruta al directorio que contiene los archivos JSON con datos de posición
                 (opcional, si no se proporciona, se estimarán las posiciones)
        use_cuda: Indicador de si se dispone de CUDA para utilizar instant-ngp en vez de NeRF puro
    
    Returns:
        Un diccionario con las rutas a las imágenes y sus correspondientes archivos de posición
    """
    # Verificar que el directorio de imágenes exista
    if not os.path.isdir(img_path):
        print(f"Error: El directorio de imágenes '{img_path}' no existe.")
        sys.exit(1)
    
    # Obtener lista de archivos de imágenes
    img_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    
    # Verificar que haya imágenes
    if not img_files:
        print(f"Error: No se encontraron imágenes en '{img_path}'.")
        sys.exit(1)
    
    # Si no se proporciona una ruta de posiciones, estimarlas todas
    if pos_path is None:
        print("No se proporcionó directorio de posiciones. Se estimarán todas las posiciones.")
        pos_path = estimate_camera_positions(img_path)
    else:
        # Verificar que el directorio de posiciones exista
        if not os.path.isdir(pos_path):
            print(f"Error: El directorio de posiciones '{pos_path}' no existe.")
            sys.exit(1)
    
    # Obtener lista de archivos de posiciones
    pos_files = [f for f in os.listdir(pos_path) if f.endswith('.json') and os.path.isfile(os.path.join(pos_path, f))]
    
    # Verificar que haya archivos JSON
    if not pos_files:
        print(f"Error: No se encontraron archivos JSON en '{pos_path}'.")
        print("Se estimarán todas las posiciones.")
        pos_path = estimate_camera_positions(img_path)
        pos_files = [f for f in os.listdir(pos_path) if f.endswith('.json') and os.path.isfile(os.path.join(pos_path, f))]
    
    # Extraer los nombres base (sin extensión) para hacer la correspondencia
    img_bases = [os.path.splitext(f)[0] for f in img_files]
    pos_bases = [os.path.splitext(f)[0] for f in pos_files]
    
    # Encontrar coincidencias entre imágenes y posiciones
    matches = set(img_bases).intersection(set(pos_bases))
    missing_positions = set(img_bases) - set(pos_bases)
    
    if not matches and not missing_positions:
        print("Error: No se encontraron imágenes válidas para procesar.")
        sys.exit(1)
    
    # Si hay imágenes sin posiciones, estimarlas
    if missing_positions:
        print(f"Se encontraron {len(missing_positions)} imágenes sin datos de posición.")
        print("Estimando posiciones para las imágenes faltantes...")
        
        # TODO: Implementar la estimación de posiciones solo para las imágenes faltantes
        missing_img_paths = [os.path.join(img_path, f + os.path.splitext(img_files[img_bases.index(f)])[1]) 
                           for f in missing_positions if f in img_bases]
        
        # Crear un directorio temporal para las imágenes sin posiciones
        temp_img_dir = os.path.join(os.path.dirname(img_path), "temp_missing_images")
        os.makedirs(temp_img_dir, exist_ok=True)
        
        # TODO: Copiar las imágenes faltantes al directorio temporal
        # for img in missing_img_paths:
        #     shutil.copy(img, temp_img_dir)
        
        # TODO: Estimar posiciones solo para estas imágenes
        # estimate_camera_positions(temp_img_dir, pos_path)
        
        # TODO: Actualizar la lista de archivos de posiciones después de estimar las faltantes
        # pos_files = [f for f in os.listdir(pos_path) if f.endswith('.json')]
        # pos_bases = [os.path.splitext(f)[0] for f in pos_files]
        
        # TODO: Limpiar el directorio temporal
        # shutil.rmtree(temp_img_dir)
        
        print("Estimación de posiciones faltantes completada.")
        
        # Actualizar la lista de coincidencias
        matches = set(img_bases).intersection(set(pos_bases))
    
    print(f"Se procesarán {len(matches)} pares de imágenes y datos de posición.")
    
    # Preparar un diccionario con las rutas a las imágenes y sus correspondientes archivos de posición
    processed_data = {}
    for base_name in matches:
        img_file = next(f for f in img_files if os.path.splitext(f)[0] == base_name)
        pos_file = next(f for f in pos_files if os.path.splitext(f)[0] == base_name)
        
        img_full_path = os.path.join(img_path, img_file)
        pos_full_path = os.path.join(pos_path, pos_file)
        
        processed_data[base_name] = {
            'image_path': img_full_path,
            'position_path': pos_full_path,
            'use_cuda': use_cuda  # Añadir la información sobre CUDA para cada par imagen-posición
        }
        
        renderer_type = "instant-ngp" if use_cuda else "NeRF puro"
        print(f"Procesando: Imagen: {img_full_path}, Posición: {pos_full_path}, Renderer: {renderer_type}")
    
    return processed_data


def load_position_data(json_path):
    """
    Carga los datos de posición desde un archivo JSON.
    
    Args:
        json_path: Ruta al archivo JSON con datos de posición
        
    Returns:
        Diccionario con los datos de posición
    """
    try:
        with open(json_path, 'r') as f:
            position_data = json.load(f)
        return position_data
    except Exception as e:
        print(f"Error al cargar datos de posición de {json_path}: {e}")
        return None