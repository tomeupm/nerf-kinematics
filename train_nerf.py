# import tensorflow as tf
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import os
# import time
# from tqdm import tqdm

# # from nerf_model import NeRF, positional_encoding, get_rays, sample_pdf, raw2outputsx

# def create_nerf_model(args):
#     """
#     Crea el modelo NeRF con los parámetros especificados.
    
#     Args:
#         args: Argumentos de configuración
        
#     Returns:
#         model: Modelo NeRF
#         model_fine: Modelo NeRF fino (para muestreo jerárquico)
#         embed_fn: Función de codificación posicional para coordenadas
#         embeddirs_fn: Función de codificación posicional para direcciones
#     """
#     # Parámetros de codificación posicional
#     multires = getattr(args, 'multires', 10)
#     multires_views = getattr(args, 'multires_views', 4)
    
#     # Crear funciones de codificación posicional
#     embed_fn = lambda x: positional_encoding(x, multires)
#     embeddirs_fn = lambda x: positional_encoding(x, multires_views)
    
#     # Calcular dimensiones de entrada
#     input_ch = 3 + 3 * 2 * multires
#     input_ch_views = 3 + 3 * 2 * multires_views
    
#     # Crear modelo principal
#     model = NeRF(D=getattr(args, 'netdepth', 8),
#                  W=getattr(args, 'netwidth', 256),
#                  input_ch=input_ch,
#                  input_ch_views=input_ch_views,
#                  skips=[4],
#                  use_viewdirs=getattr(args, 'use_viewdirs', True))
    
#     # Crear modelo fino para muestreo jerárquico
#     model_fine = NeRF(D=getattr(args, 'netdepth_fine', 8),
#                       W=getattr(args, 'netwidth_fine', 256),
#                       input_ch=input_ch,
#                       input_ch_views=input_ch_views,
#                       skips=[4],
#                       use_viewdirs=getattr(args, 'use_viewdirs', True))
    
#     return model, model_fine, embed_fn, embeddirs_fn


# def render_rays(ray_batch, network_fn, network_fine, N_samples, embed_fn, embeddirs_fn,
#                 near=0., far=1., N_importance=0):
#     """
#     Renderiza un batch de rayos usando los modelos NeRF.
    
#     Args:
#         ray_batch: Batch de rayos [N_rays, ro+rd] = [N_rays, 3+3=6]
#         network_fn: Función del modelo principal
#         network_fine: Función del modelo fino
#         N_samples: Número de muestras por rayo
#         embed_fn: Función de embedding para coordenadas
#         embeddirs_fn: Función de embedding para direcciones
#         near: Distancia cercana
#         far: Distancia lejana
#         N_importance: Número de muestras adicionales para muestreo fino
        
#     Returns:
#         Diccionario con mapas RGB, profundidad, etc.
#     """
#     N_rays = ray_batch.shape[0]
#     rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] cada uno
#     viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
#     # Usar bounds de los parámetros near/far en lugar de extraer del ray_batch
#     device = ray_batch.device
    
#     # Muestreo estratificado
#     t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
#     z_vals = near * (1. - t_vals) + far * t_vals
#     z_vals = z_vals.expand([N_rays, N_samples])
    
#     # Añadir perturbación al muestreo
#     mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
#     upper = torch.cat([mids, z_vals[..., -1:]], -1)
#     lower = torch.cat([z_vals[..., :1], mids], -1)
#     t_rand = torch.rand(z_vals.shape, device=device)
#     z_vals = lower + (upper - lower) * t_rand
    
#     # Obtener puntos 3D
#     pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    
#     # Renderizar con modelo principal
#     raw = run_network(pts, viewdirs, network_fn, embed_fn, embeddirs_fn)
#     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)
    
#     ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    
#     # Muestreo jerárquico con modelo fino
#     if N_importance > 0:
#         ret0 = ret
        
#         # Muestreo por importancia
#         z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
#         z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=False)
#         z_samples = z_samples.detach()
        
#         # Combinar muestras
#         z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
#         pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
#         # Renderizar con modelo fino
#         raw = run_network(pts, viewdirs, network_fine, embed_fn, embeddirs_fn)
#         rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)
        
#         ret['rgb_map'] = rgb_map
#         ret['disp_map'] = disp_map
#         ret['acc_map'] = acc_map
#         ret['rgb0'] = ret0['rgb_map']
#         ret['disp0'] = ret0['disp_map']
#         ret['acc0'] = ret0['acc_map']
#         ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
    
#     return ret


# def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
#     """
#     Ejecuta la red en chunks para evitar out-of-memory.
#     """
#     inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
#     embedded = embed_fn(inputs_flat)
    
#     if viewdirs is not None:
#         input_dirs = viewdirs[:, None].expand(inputs.shape)
#         input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
#         embedded_dirs = embeddirs_fn(input_dirs_flat)
#         embedded = torch.cat([embedded, embedded_dirs], -1)
    
#     outputs_flat = batchify(fn, netchunk)(embedded)
#     outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
#     return outputs


# def batchify(fn, chunk):
#     """Función auxiliar para procesar en chunks."""
#     if chunk is None:
#         return fn
#     def ret(inputs):
#         return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
#     return ret


# def train_nerf(args, imgs, poses, hwf, i_split, render_poses=None):
#     """
#     Función principal de entrenamiento de NeRF.
    
#     Args:
#         args: Argumentos de configuración
#         imgs: Imágenes de entrenamiento [N, H, W, 3]
#         poses: Poses de cámara [N, 4, 4]
#         hwf: [Height, Width, focal]
#         i_split: Índices [train, val, test]
#         render_poses: Poses para renderizado
#     """
#     print("Iniciando entrenamiento de NeRF...")
    
#     # Configuración del dispositivo con soporte para Apple Silicon
#     device = torch.device("cpu")
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")  # Apple Silicon GPU
#     print(f"Usando dispositivo: {device}")
    
#     # Extraer parámetros de imagen
#     H, W, focal = hwf
#     i_train, i_val, i_test = i_split
    
#     # Convertir a tensores
#     imgs = torch.tensor(imgs).to(device)
#     poses = torch.tensor(poses).to(device)
    
#     # Crear modelos
#     model, model_fine, embed_fn, embeddirs_fn = create_nerf_model(args)
#     model.to(device)
#     model_fine.to(device)
    
#     # Optimizador
#     optimizer = optim.Adam([
#         {'params': model.parameters(), 'lr': getattr(args, 'lrate', 5e-4)},
#         {'params': model_fine.parameters(), 'lr': getattr(args, 'lrate', 5e-4)}
#     ])
    
#     # Parámetros de entrenamiento
#     N_iters = getattr(args, 'N_iters', 200000)
#     N_samples = getattr(args, 'N_samples', 64)
#     N_importance = getattr(args, 'N_importance', 128)
#     chunk = getattr(args, 'chunk', 1024*32)
    
#     # Bucle de entrenamiento
#     start = time.time()
    
#     for i in tqdm(range(N_iters), desc="Entrenando NeRF"):
#         # Seleccionar imagen aleatoria de entrenamiento
#         img_i = np.random.choice(i_train)
#         target = imgs[img_i].to(device)
#         pose = poses[img_i, :3, :4].to(device)
        
#         # Generar rayos
#         rays_o, rays_d = get_rays(H, W, focal, pose)
        
#         # Seleccionar subset de rayos para el batch
#         coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H, device=device), torch.linspace(0, W-1, W, device=device), indexing='ij'), -1)
#         coords = torch.reshape(coords, [-1, 2])
#         select_inds = np.random.choice(coords.shape[0], size=[getattr(args, 'N_rand', 1024)], replace=False)
#         select_coords = coords[select_inds].long()
        
#         rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
#         rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        
#         # Concatenar rayos en formato [N_rays, 6] donde 6 = [rays_o + rays_d]
#         rays = torch.cat([rays_o, rays_d], -1)  # [N_rays, 6]
#         target_s = target[select_coords[:, 0], select_coords[:, 1]][:, :3]  # Solo RGB, no alfa
        
#         # Obtener bounds
#         near = getattr(args, 'near', 2.0)
#         far = getattr(args, 'far', 6.0)
        
#         # Renderizar
#         ret = render_rays(rays, lambda x: model(x), lambda x: model_fine(x),
#                          N_samples, embed_fn, embeddirs_fn,
#                          near=near, far=far, N_importance=N_importance)
        
#         rgb = ret['rgb_map']
        
#         # Calcular pérdida
#         img_loss = torch.mean((rgb - target_s) ** 2)
#         loss = img_loss
        
#         if 'rgb0' in ret:
#             img_loss0 = torch.mean((ret['rgb0'] - target_s) ** 2)
#             loss = loss + img_loss0
        
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Logging
#         if i % getattr(args, 'i_print', 100) == 0:
#             tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")
        
#         # Guardar checkpoints
#         if i % getattr(args, 'i_weights', 10000) == 0:
#             path = os.path.join(getattr(args, 'basedir', './logs'), 'checkpoints', f'{i:06d}.tar')
#             os.makedirs(os.path.dirname(path), exist_ok=True)
#             torch.save({
#                 'global_step': i,
#                 'network_fn_state_dict': model.state_dict(),
#                 'network_fine_state_dict': model_fine.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }, path)
#             print(f'Guardado checkpoint en {path}')
    
#     print("Entrenamiento completado!")


# class TrainingArgs:
#     """Clase simple para argumentos de entrenamiento con valores por defecto."""
#     def __init__(self):
#         # Arquitectura del modelo
#         self.netdepth = 8
#         self.netwidth = 256
#         self.netdepth_fine = 8
#         self.netwidth_fine = 256
#         self.use_viewdirs = True
        
#         # Codificación posicional
#         self.multires = 10
#         self.multires_views = 4
        
#         # Entrenamiento
#         self.N_iters = 200000
#         self.lrate = 5e-4
#         self.N_samples = 64
#         self.N_importance = 128
#         self.N_rand = 1024
#         self.chunk = 1024*32
        
#         # Geometría de la escena
#         self.near = 2.0
#         self.far = 6.0
        
#         # Logging
#         self.i_print = 100
#         self.i_weights = 10000
#         self.basedir = './logs'
