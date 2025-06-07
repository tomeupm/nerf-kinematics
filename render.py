import tensorflow as tf
import torch
import numpy as np
import os
import time
import imageio
from tqdm import tqdm

# from nerf_model import NeRF, positional_encoding, get_rays, raw2outputs
from nerf_model import batchify, get_rays


# --------------------------------Versión Original-------------------------------

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret



def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Convert inputs to float32 for consistency
    H = tf.cast(H, tf.float32)
    W = tf.cast(W, tf.float32)
    focal = tf.cast(focal, tf.float32)
    near = tf.cast(near, tf.float32)
    
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d
  
  
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = tf.split(rays, 2, axis=0)
        rays_o = tf.squeeze(rays_o, axis=0)
        rays_d = tf.squeeze(rays_d, axis=0)

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    
    # Create near and far tensors with the same shape as rays
    rays_shape = tf.shape(rays_d)[0]
    near_expanded = tf.fill([rays_shape, 1], tf.cast(near, tf.float32))
    far_expanded = tf.fill([rays_shape, 1], tf.cast(far, tf.float32))

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near_expanded, far_expanded], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]







# --------------------------------Versión Copilot--------------------------------

# def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
#     """
#     Renderiza imágenes para un conjunto de poses.
    
#     Args:
#         render_poses: Poses de renderizado [N, 4, 4]
#         hwf: [Height, Width, focal]
#         chunk: Tamaño del chunk para procesamiento
#         render_kwargs: Argumentos para la función de renderizado
#         gt_imgs: Imágenes ground truth (opcional)
#         savedir: Directorio para guardar imágenes
#         render_factor: Factor de downsampling (0 = sin downsampling)
        
#     Returns:
#         rgbs: Lista de imágenes renderizadas
#         disps: Lista de mapas de disparidad
#     """
#     H, W, focal = hwf
    
#     if render_factor != 0:
#         # Reducir resolución
#         H = H // render_factor
#         W = W // render_factor
#         focal = focal / render_factor
    
#     rgbs = []
#     disps = []
    
#     t = time.time()
#     for i, c2w in enumerate(tqdm(render_poses)):
#         print(f"Renderizando {i+1}/{len(render_poses)}")
        
#         rays_o, rays_d = get_rays(H, W, focal, c2w[:3, :4])
#         rays_o = rays_o.reshape([-1, 3])
#         rays_d = rays_d.reshape([-1, 3])
        
#         # Añadir bounds
#         sh = rays_d.shape
#         rays_o = rays_o.reshape([-1, 3])
#         rays_d = rays_d.reshape([-1, 3])
        
#         near = render_kwargs.get('near', 0.)
#         far = render_kwargs.get('far', 1.)
#         rays = torch.cat([rays_o, rays_d, 
#                          near * torch.ones_like(rays_o[..., :1]),
#                          far * torch.ones_like(rays_o[..., :1])], -1)
        
#         # Renderizar en chunks
#         all_ret = {}
#         for k in range(0, rays.shape[0], chunk):
#             ret = render_rays(rays[k:k+chunk], **render_kwargs)
#             for key in ret:
#                 if key not in all_ret:
#                     all_ret[key] = []
#                 all_ret[key].append(ret[key])
        
#         # Concatenar resultados
#         all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        
#         # Reshape a imagen
#         for k in all_ret:
#             k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
#             all_ret[k] = torch.reshape(all_ret[k], k_sh)
        
#         rgb = all_ret['rgb_map'].cpu().numpy()
#         disp = all_ret['disp_map'].cpu().numpy()
        
#         rgbs.append(rgb)
#         disps.append(disp)
        
#         # Guardar imagen si se especifica directorio
#         if savedir is not None:
#             rgb8 = to8b(rgb)
#             filename = os.path.join(savedir, f'{i:03d}.png')
#             imageio.imwrite(filename, rgb8)
    
#     rgbs = np.array(rgbs)
#     disps = np.array(disps)
    
#     return rgbs, disps


# def render_novel_views(args, imgs, poses, hwf, i_split, render_poses):
#     """
#     Renderiza vistas noveles usando un modelo NeRF entrenado.
    
#     Args:
#         args: Argumentos de configuración
#         imgs: Imágenes originales
#         poses: Poses de cámara
#         hwf: [Height, Width, focal]
#         i_split: División de datos
#         render_poses: Poses para renderizado
#     """
#     print("Renderizando vistas noveles...")
    
#     # Configuración del dispositivo con soporte para Apple Silicon
#     device = torch.device("cpu")
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")  # Apple Silicon GPU
#     print(f"Usando dispositivo: {device}")
    
#     # Crear modelos
#     model, model_fine, embed_fn, embeddirs_fn = create_nerf_model(args)
#     model.to(device)
#     model_fine.to(device)
    
#     # Cargar checkpoint si existe
#     ckpt_path = getattr(args, 'ft_path', None)
#     if ckpt_path is not None and os.path.exists(ckpt_path):
#         print(f"Cargando checkpoint desde {ckpt_path}")
#         ckpt = torch.load(ckpt_path)
#         model.load_state_dict(ckpt['network_fn_state_dict'])
#         model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
#     # Configurar argumentos de renderizado
#     render_kwargs = {
#         'network_fn': lambda x: model(x),
#         'network_fine': lambda x: model_fine(x),
#         'N_samples': getattr(args, 'N_samples', 64),
#         'embed_fn': embed_fn,
#         'embeddirs_fn': embeddirs_fn,
#         'near': getattr(args, 'near', 2.0),
#         'far': getattr(args, 'far', 6.0),
#         'N_importance': getattr(args, 'N_importance', 128),
#     }
    
#     # Crear directorio de salida
#     testsavedir = os.path.join(getattr(args, 'basedir', './logs'), 'renderonly_test')
#     os.makedirs(testsavedir, exist_ok=True)
    
#     # Renderizar
#     H, W, focal = hwf
#     chunk = getattr(args, 'chunk', 1024*32)
    
#     with torch.no_grad():
#         rgbs, disps = render_path(torch.tensor(render_poses, dtype=torch.float32).to(device), hwf, chunk, render_kwargs, 
#                                 savedir=testsavedir)
    
#     print(f"Renderizado completado. Imágenes guardadas en {testsavedir}")
    
#     # Crear video si se especifica
#     if getattr(args, 'render_video', False):
#         create_video(testsavedir, fps=getattr(args, 'fps', 30))


# def create_video(image_dir, fps=30, video_name='render'):
#     """
#     Crea un video a partir de las imágenes renderizadas.
    
#     Args:
#         image_dir: Directorio con las imágenes
#         fps: Frames por segundo
#         video_name: Nombre del video de salida
#     """
#     print(f"Creando video desde {image_dir}...")
    
#     # Obtener lista de imágenes
#     images = []
#     filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
#     for filename in filenames:
#         img_path = os.path.join(image_dir, filename)
#         img = imageio.imread(img_path)
#         images.append(img)
    
#     # Crear video
#     video_path = os.path.join(image_dir, f'{video_name}.mp4')
#     imageio.mimwrite(video_path, images, fps=fps, quality=8)
#     print(f"Video guardado en {video_path}")


# def extract_mesh(args, imgs, poses, hwf, resolution=512):
#     """
#     Extrae una malla 3D usando marching cubes.
    
#     Args:
#         args: Argumentos de configuración
#         imgs: Imágenes originales
#         poses: Poses de cámara
#         hwf: [Height, Width, focal]
#         resolution: Resolución de la grilla 3D
#     """
#     print("Extrayendo malla 3D...")
    
#     # Configuración del dispositivo con soporte para Apple Silicon
#     device = torch.device("cpu")
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")  # Apple Silicon GPU
#     print(f"Usando dispositivo: {device}")
    
#     # Crear modelos
#     model, model_fine, embed_fn, embeddirs_fn = create_nerf_model(args)
#     model.to(device)
#     model_fine.to(device)
    
#     # Cargar checkpoint
#     ckpt_path = getattr(args, 'ft_path', None)
#     if ckpt_path is not None and os.path.exists(ckpt_path):
#         print(f"Cargando checkpoint desde {ckpt_path}")
#         ckpt = torch.load(ckpt_path)
#         model.load_state_dict(ckpt['network_fn_state_dict'])
#         model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
#     # Definir bounding box de la escena
#     bounds = np.array([[-1., -1., -1.], [1., 1., 1.]])
    
#     # Crear grilla 3D
#     x = np.linspace(bounds[0, 0], bounds[1, 0], resolution)
#     y = np.linspace(bounds[0, 1], bounds[1, 1], resolution)
#     z = np.linspace(bounds[0, 2], bounds[1, 2], resolution)
    
#     xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
#     points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
#     # Evaluar densidad en todos los puntos
#     chunk_size = 1024 * 64
#     densities = []
    
#     with torch.no_grad():
#         for i in tqdm(range(0, len(points), chunk_size), desc="Evaluando densidad"):
#             chunk_points = torch.tensor(points[i:i+chunk_size], dtype=torch.float32).to(device)
            
#             # Codificación posicional
#             embedded = embed_fn(chunk_points)
            
#             # Dummy view directions (no afectan la densidad)
#             viewdirs = torch.zeros_like(chunk_points)
#             embedded_dirs = embeddirs_fn(viewdirs)
#             embedded = torch.cat([embedded, embedded_dirs], -1)
            
#             # Evaluar modelo
#             raw_output = model_fine(embedded)
#             density = torch.relu(raw_output[..., 3])
#             densities.append(density.cpu().numpy())
    
#     densities = np.concatenate(densities, axis=0)
#     densities = densities.reshape(resolution, resolution, resolution)
    
#     # Aplicar marching cubes
#     try:
#         from skimage import measure
        
#         threshold = getattr(args, 'mesh_threshold', 50.0)
#         verts, faces, _, _ = measure.marching_cubes(densities, threshold)
        
#         # Escalar vértices al espacio original
#         verts[:, 0] = (verts[:, 0] / resolution) * (bounds[1, 0] - bounds[0, 0]) + bounds[0, 0]
#         verts[:, 1] = (verts[:, 1] / resolution) * (bounds[1, 1] - bounds[0, 1]) + bounds[0, 1]
#         verts[:, 2] = (verts[:, 2] / resolution) * (bounds[1, 2] - bounds[0, 2]) + bounds[0, 2]
        
#         # Guardar malla
#         mesh_path = os.path.join(getattr(args, 'basedir', './logs'), 'mesh.ply')
#         save_mesh_ply(verts, faces, mesh_path)
#         print(f"Malla guardada en {mesh_path}")
        
#     except ImportError:
#         print("Error: scikit-image no está instalado. No se puede extraer la malla.")
#         print("Instale con: pip install scikit-image")


# def save_mesh_ply(vertices, faces, filename):
#     """
#     Guarda una malla en formato PLY.
    
#     Args:
#         vertices: Vértices de la malla [N, 3]
#         faces: Caras de la malla [M, 3]
#         filename: Nombre del archivo de salida
#     """
#     with open(filename, 'w') as f:
#         f.write("ply\n")
#         f.write("format ascii 1.0\n")
#         f.write(f"element vertex {len(vertices)}\n")
#         f.write("property float x\n")
#         f.write("property float y\n")
#         f.write("property float z\n")
#         f.write(f"element face {len(faces)}\n")
#         f.write("property list uchar int vertex_indices\n")
#         f.write("end_header\n")
        
#         for vertex in vertices:
#             f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
#         for face in faces:
#             f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# def to8b(x):
#     """Convierte imagen float a uint8."""
#     return (255*np.clip(x, 0, 1)).astype(np.uint8)
