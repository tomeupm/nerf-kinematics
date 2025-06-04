import argparse
import os
import sys

from data_loader import load_nerf_data
from train_nerf import train_nerf, TrainingArgs
from render import render_novel_views, extract_mesh


def main():
    """Función principal del programa."""
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description='Herramienta para reconstrucción 3D usando NeRFs a partir de imágenes y datos de posición.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Argumentos principales
    parser.add_argument('--img', metavar='PATH', type=str, required=True,
                        help='Ruta al directorio que contiene las imágenes')
    parser.add_argument('--pos', metavar='PATH', type=str, required=True,
                        help='Ruta al archivo poses.txt que contiene las matrices de transformación')
    parser.add_argument('--half_res', action='store_true',
                        help='Reducir la resolución de las imágenes a la mitad para acelerar el entrenamiento')
    
    # Argumentos de entrenamiento
    parser.add_argument('--train', action='store_true',
                        help='Entrenar el modelo NeRF')
    parser.add_argument('--render', action='store_true',
                        help='Renderizar vistas noveles usando modelo entrenado')
    parser.add_argument('--extract_mesh', action='store_true',
                        help='Extraer malla 3D del modelo entrenado')
    
    # Argumentos de configuración
    parser.add_argument('--config', type=str,
                        help='Ruta al archivo de configuración (opcional)')
    parser.add_argument('--basedir', type=str, default='./logs',
                        help='Directorio base para logs y checkpoints')
    parser.add_argument('--expname', type=str, default='nerf_experiment',
                        help='Nombre del experimento')
    
    # Argumentos del modelo
    parser.add_argument('--N_iters', type=int, default=200000,
                        help='Número de iteraciones de entrenamiento')
    parser.add_argument('--lrate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='Número de muestras por rayo (coarse)')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='Número de muestras adicionales por rayo (fine)')
    
    # Parsear los argumentos
    args = parser.parse_args()
    
    # Si no se especifica ninguna acción, entrenar por defecto
    if not (args.train or args.render or args.extract_mesh):
        args.train = True
    
    # Crear directorio de experimento
    args.basedir = os.path.join(args.basedir, args.expname)
    os.makedirs(args.basedir, exist_ok=True)
    
    print("="*50)
    print("NEURAL RADIANCE FIELDS (NeRF) - Reconstrucción 3D")
    print("="*50)
    
    # Cargar datos
    print("Cargando datos para NeRF...")
    imgs, poses, render_poses, hwf, i_split = load_nerf_data(args.img, args.pos, half_res=args.half_res)
    
    # Extraer información de los datos cargados
    H, W, focal = hwf
    i_train, i_val, i_test = i_split
    
    print(f"Datos cargados exitosamente:")
    print(f"  - Imágenes: {imgs.shape}")
    print(f"  - Poses: {poses.shape}")
    print(f"  - Resolución: {H}x{W}, focal: {focal:.2f}")
    print(f"  - Train/Val/Test: {len(i_train)}/{len(i_val)}/{len(i_test)}")
    print(f"  - Render poses: {len(render_poses) if render_poses is not None else 0}")
    
    # Crear configuración de entrenamiento
    train_args = TrainingArgs()
    
    # Actualizar configuración con argumentos de línea de comandos
    for key, value in vars(args).items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    
    # Ejecutar acciones solicitadas
    if args.train:
        print("\n" + "="*50)
        print("ENTRENANDO MODELO NERF")
        print("="*50)
        train_nerf(train_args, imgs, poses, hwf, i_split, render_poses)
    
    if args.render:
        print("\n" + "="*50)
        print("RENDERIZANDO VISTAS NOVELES")
        print("="*50)
        if render_poses is not None:
            render_novel_views(train_args, imgs, poses, hwf, i_split, render_poses)
        else:
            print("Error: No hay poses de renderizado disponibles.")
    
    if args.extract_mesh:
        print("\n" + "="*50)
        print("EXTRAYENDO MALLA 3D")
        print("="*50)
        extract_mesh(train_args, imgs, poses, hwf)
    
    print("\n" + "="*50)
    print("PROCESO COMPLETADO")
    print("="*50)


# Ejecutar el programa cuando se llame como script principal
if __name__ == '__main__':
    main()
