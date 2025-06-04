import torch
import numpy as np
import os
import time
import imageio
from tqdm import tqdm

from nerf_model import NeRF, positional_encoding, get_rays, raw2outputs
from train_nerf import run_network, render_rays, create_nerf_model


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """
    Renderiza imágenes para un conjunto de poses.
    
    Args:
        render_poses: Poses de renderizado [N, 4, 4]
        hwf: [Height, Width, focal]
        chunk: Tamaño del chunk para procesamiento
        render_kwargs: Argumentos para la función de renderizado
        gt_imgs: Imágenes ground truth (opcional)
        savedir: Directorio para guardar imágenes
        render_factor: Factor de downsampling (0 = sin downsampling)
        
    Returns:
        rgbs: Lista de imágenes renderizadas
        disps: Lista de mapas de disparidad
    """
    H, W, focal = hwf
    
    if render_factor != 0:
        # Reducir resolución
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
    
    rgbs = []
    disps = []
    
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(f"Renderizando {i+1}/{len(render_poses)}")
        
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4])
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])
        
        # Añadir bounds
        sh = rays_d.shape
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])
        
        near = render_kwargs.get('near', 0.)
        far = render_kwargs.get('far', 1.)
        rays = torch.cat([rays_o, rays_d, 
                         near * torch.ones_like(rays_o[..., :1]),
                         far * torch.ones_like(rays_o[..., :1])], -1)
        
        # Renderizar en chunks
        all_ret = {}
        for k in range(0, rays.shape[0], chunk):
            ret = render_rays(rays[k:k+chunk], **render_kwargs)
            for key in ret:
                if key not in all_ret:
                    all_ret[key] = []
                all_ret[key].append(ret[key])
        
        # Concatenar resultados
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        
        # Reshape a imagen
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
        
        rgb = all_ret['rgb_map'].cpu().numpy()
        disp = all_ret['disp_map'].cpu().numpy()
        
        rgbs.append(rgb)
        disps.append(disp)
        
        # Guardar imagen si se especifica directorio
        if savedir is not None:
            rgb8 = to8b(rgb)
            filename = os.path.join(savedir, f'{i:03d}.png')
            imageio.imwrite(filename, rgb8)
    
    rgbs = np.array(rgbs)
    disps = np.array(disps)
    
    return rgbs, disps


def render_novel_views(args, imgs, poses, hwf, i_split, render_poses):
    """
    Renderiza vistas noveles usando un modelo NeRF entrenado.
    
    Args:
        args: Argumentos de configuración
        imgs: Imágenes originales
        poses: Poses de cámara
        hwf: [Height, Width, focal]
        i_split: División de datos
        render_poses: Poses para renderizado
    """
    print("Renderizando vistas noveles...")
    
    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear modelos
    model, model_fine, embed_fn, embeddirs_fn = create_nerf_model(args)
    model.to(device)
    model_fine.to(device)
    
    # Cargar checkpoint si existe
    ckpt_path = getattr(args, 'ft_path', None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Cargando checkpoint desde {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
    # Configurar argumentos de renderizado
    render_kwargs = {
        'network_fn': lambda x: model(x),
        'network_fine': lambda x: model_fine(x),
        'N_samples': getattr(args, 'N_samples', 64),
        'embed_fn': embed_fn,
        'embeddirs_fn': embeddirs_fn,
        'near': getattr(args, 'near', 0.),
        'far': getattr(args, 'far', 1.),
        'N_importance': getattr(args, 'N_importance', 128),
        'raw_noise_std': 0.,
    }
    
    # Crear directorio de salida
    testsavedir = os.path.join(getattr(args, 'basedir', './logs'), 'renderonly_test')
    os.makedirs(testsavedir, exist_ok=True)
    
    # Renderizar
    H, W, focal = hwf
    chunk = getattr(args, 'chunk', 1024*32)
    
    with torch.no_grad():
        rgbs, disps = render_path(torch.tensor(render_poses).to(device), hwf, chunk, render_kwargs, 
                                savedir=testsavedir)
    
    print(f"Renderizado completado. Imágenes guardadas en {testsavedir}")
    
    # Crear video si se especifica
    if getattr(args, 'render_video', False):
        create_video(testsavedir, fps=getattr(args, 'fps', 30))


def create_video(image_dir, fps=30, video_name='render'):
    """
    Crea un video a partir de las imágenes renderizadas.
    
    Args:
        image_dir: Directorio con las imágenes
        fps: Frames por segundo
        video_name: Nombre del video de salida
    """
    print(f"Creando video desde {image_dir}...")
    
    # Obtener lista de imágenes
    images = []
    filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        img = imageio.imread(img_path)
        images.append(img)
    
    # Crear video
    video_path = os.path.join(image_dir, f'{video_name}.mp4')
    imageio.mimwrite(video_path, images, fps=fps, quality=8)
    print(f"Video guardado en {video_path}")


def extract_mesh(args, imgs, poses, hwf, resolution=512):
    """
    Extrae una malla 3D usando marching cubes.
    
    Args:
        args: Argumentos de configuración
        imgs: Imágenes originales
        poses: Poses de cámara
        hwf: [Height, Width, focal]
        resolution: Resolución de la grilla 3D
    """
    print("Extrayendo malla 3D...")
    
    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear modelos
    model, model_fine, embed_fn, embeddirs_fn = create_nerf_model(args)
    model.to(device)
    model_fine.to(device)
    
    # Cargar checkpoint
    ckpt_path = getattr(args, 'ft_path', None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Cargando checkpoint desde {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
    # Definir bounding box de la escena
    bounds = np.array([[-1., -1., -1.], [1., 1., 1.]])
    
    # Crear grilla 3D
    x = np.linspace(bounds[0, 0], bounds[1, 0], resolution)
    y = np.linspace(bounds[0, 1], bounds[1, 1], resolution)
    z = np.linspace(bounds[0, 2], bounds[1, 2], resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Evaluar densidad en todos los puntos
    chunk_size = 1024 * 64
    densities = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(points), chunk_size), desc="Evaluando densidad"):
            chunk_points = torch.tensor(points[i:i+chunk_size]).float().to(device)
            
            # Codificación posicional
            embedded = embed_fn(chunk_points)
            
            # Dummy view directions (no afectan la densidad)
            viewdirs = torch.zeros_like(chunk_points)
            embedded_dirs = embeddirs_fn(viewdirs)
            embedded = torch.cat([embedded, embedded_dirs], -1)
            
            # Evaluar modelo
            raw_output = model_fine(embedded)
            density = torch.relu(raw_output[..., 3])
            densities.append(density.cpu().numpy())
    
    densities = np.concatenate(densities, axis=0)
    densities = densities.reshape(resolution, resolution, resolution)
    
    # Aplicar marching cubes
    try:
        from skimage import measure
        
        threshold = getattr(args, 'mesh_threshold', 50.0)
        verts, faces, _, _ = measure.marching_cubes(densities, threshold)
        
        # Escalar vértices al espacio original
        verts[:, 0] = (verts[:, 0] / resolution) * (bounds[1, 0] - bounds[0, 0]) + bounds[0, 0]
        verts[:, 1] = (verts[:, 1] / resolution) * (bounds[1, 1] - bounds[0, 1]) + bounds[0, 1]
        verts[:, 2] = (verts[:, 2] / resolution) * (bounds[1, 2] - bounds[0, 2]) + bounds[0, 2]
        
        # Guardar malla
        mesh_path = os.path.join(getattr(args, 'basedir', './logs'), 'mesh.ply')
        save_mesh_ply(verts, faces, mesh_path)
        print(f"Malla guardada en {mesh_path}")
        
    except ImportError:
        print("Error: scikit-image no está instalado. No se puede extraer la malla.")
        print("Instale con: pip install scikit-image")


def save_mesh_ply(vertices, faces, filename):
    """
    Guarda una malla en formato PLY.
    
    Args:
        vertices: Vértices de la malla [N, 3]
        faces: Caras de la malla [M, 3]
        filename: Nombre del archivo de salida
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def to8b(x):
    """Convierte imagen float a uint8."""
    return (255*np.clip(x, 0, 1)).astype(np.uint8)
