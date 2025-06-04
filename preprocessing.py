import os
import sys
import json
import shutil
import numpy as np
import re


def parse_poses_file(poses_file_path):
    """
    Parsea el archivo poses.txt y extrae las matrices de transformación.
    
    Args:
        poses_file_path: Ruta al archivo poses.txt
        
    Returns:
        Lista de matrices de transformación 4x4 como arrays de numpy
    """
    poses = []
    
    try:
        with open(poses_file_path, 'r') as f:
            content = f.read()
        
        # Buscar todas las matrices en el formato [... ; ... ; ... ; ...]
        matrix_pattern = r'\[\s*(.*?)\s*\]'
        matrices = re.findall(matrix_pattern, content, re.DOTALL)
        
        for matrix_str in matrices:
            # Dividir por punto y coma para obtener las filas
            rows = matrix_str.split(';')
            matrix_rows = []
            
            for row in rows:
                # Limpiar la fila y dividir por espacios/comas
                row_clean = row.strip().replace(',', '')
                if row_clean:  # Ignorar filas vacías
                    # Extraer números de la fila
                    numbers = re.findall(r'-?\d+\.?\d*', row_clean)
                    if len(numbers) == 4:  # Debe tener 4 elementos por fila
                        matrix_rows.append([float(num) for num in numbers])
            
            if len(matrix_rows) == 4:  # Debe tener 4 filas
                poses.append(np.array(matrix_rows))
        
        print(f"Se han parseado {len(poses)} matrices de poses desde {poses_file_path}")
        return poses
        
    except Exception as e:
        print(f"Error al parsear el archivo de poses {poses_file_path}: {e}")
        return []






def process_images_and_positions(img_path, poses_file_path, use_cuda=False):
    """
    Procesa las imágenes y datos de posición para la reconstrucción 3D con NeRFs.
    
    Args:
        img_path: Ruta al directorio que contiene las imágenes
        poses_file_path: Ruta al archivo poses.txt (requerido)
        use_cuda: Indicador de si se dispone de CUDA para utilizar instant-ngp en vez de NeRF puro
    
    Returns:
        Un diccionario con las rutas a las imágenes y sus correspondientes datos de posición
    """
    # Verificar que el directorio de imágenes exista
    if not os.path.isdir(img_path):
        print(f"Error: El directorio de imágenes '{img_path}' no existe.")
        sys.exit(1)
    
    # Verificar que el archivo poses.txt exista
    if not os.path.isfile(poses_file_path) or not poses_file_path.endswith('poses.txt'):
        print(f"Error: Debe proporcionar un archivo poses.txt válido. Archivo proporcionado: {poses_file_path}")
        sys.exit(1)
    
    # Obtener lista de archivos de imágenes y ordenarlas por el número al final del nombre
    img_files = [f for f in os.listdir(img_path) 
                 if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Verificar que haya imágenes
    if not img_files:
        print(f"Error: No se encontraron imágenes en '{img_path}'.")
        sys.exit(1)
    
    def extract_number_from_filename(filename):
        """Extrae el número al final del nombre del archivo."""
        # Buscar números al final del nombre (antes de la extensión)
        match = re.search(r'(\d+)\.', filename)
        return int(match.group(1)) if match else float('inf')
    
    # Ordenar imágenes por el número al final del nombre
    img_files.sort(key=extract_number_from_filename)
    
    print(f"Procesando archivo de poses: {poses_file_path}")
    
    # Parsear las matrices de poses
    poses = parse_poses_file(poses_file_path)
    
    if not poses:
        print("Error: No se pudieron parsear matrices de poses del archivo.")
        sys.exit(1)
    
    print(f"Se encontraron {len(img_files)} imágenes ordenadas numéricamente.")
    print(f"Se tienen {len(poses)} matrices de poses.")
    
    # Verificar que el número de poses coincida con el número de imágenes
    if len(poses) != len(img_files):
        print(f"Error: El número de poses ({len(poses)}) no coincide con el número de imágenes ({len(img_files)})")
        print("Asegúrese de que el archivo poses.txt contenga exactamente una matriz por cada imagen.")
        sys.exit(1)
    
    print(f"Se procesarán {len(poses)} pares de imágenes y datos de posición.")
    
    # Preparar un diccionario con las rutas a las imágenes y sus correspondientes datos de posición
    processed_data = {}
    for i, img_file in enumerate(img_files):
        pose_matrix = poses[i]
        image_name = os.path.splitext(img_file)[0]  # Nombre sin extensión
        img_full_path = os.path.join(img_path, img_file)
        
        processed_data[image_name] = {
            'image_path': img_full_path,
            'pose_matrix': pose_matrix,
            'pose_index': i,
            'use_cuda': use_cuda
        }
        
        renderer_type = "instant-ngp" if use_cuda else "NeRF puro"
        print(f"Procesando: Imagen: {img_full_path}, Pose: matriz {i}, Renderer: {renderer_type}")
    
    return processed_data


