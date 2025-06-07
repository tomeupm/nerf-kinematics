import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------Versión Original-------------------------------

def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
  
  
# Model architecture
def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views,))
    
    # Usar Lambda layers para dividir los inputs
    inputs_pts = tf.keras.layers.Lambda(lambda x: x[:, :input_ch])(inputs)
    inputs_views = tf.keras.layers.Lambda(lambda x: x[:, input_ch:])(inputs)

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.keras.layers.Concatenate()([inputs_pts, outputs])

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.keras.layers.Concatenate()([bottleneck, inputs_views])
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.keras.layers.Concatenate()([outputs, alpha_out])
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
  

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret
  
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

def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d




# --------------------------------Versión Copilot--------------------------------

# class NeRF(nn.Module):
#     """
#     Implementación del modelo Neural Radiance Field (NeRF).
    
#     Red neuronal que mapea coordenadas 3D (x, y, z) y direcciones de vista (theta, phi)
#     a densidad volumétrica y color RGB.
#     """
    
#     def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
#         """
#         Args:
#             D: Número de capas en la red
#             W: Número de neuronas por capa
#             input_ch: Canales de entrada para coordenadas espaciales
#             input_ch_views: Canales de entrada para direcciones de vista
#             output_ch: Canales de salida (RGB + densidad)
#             skips: Lista de capas donde añadir conexiones residuales
#             use_viewdirs: Si usar direcciones de vista para predicción de color
#         """
#         super(NeRF, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.input_ch_views = input_ch_views
#         self.skips = skips
#         self.use_viewdirs = use_viewdirs
        
#         # Capas para procesamiento de coordenadas espaciales
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
#         )
        
#         # Capa para predecir densidad
#         self.density_linear = nn.Linear(W, 1)
        
#         # Capa de características antes del color
#         self.feature_linear = nn.Linear(W, W)
        
#         # Capas para procesamiento de direcciones de vista
#         if use_viewdirs:
#             self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
#             self.rgb_linear = nn.Linear(W//2, 3)
#         else:
#             self.rgb_linear = nn.Linear(W, 3)
            
#     def forward(self, x):
#         """
#         Forward pass del modelo NeRF.
        
#         Args:
#             x: Tensor de entrada [N, input_ch + input_ch_views]
#                Primeros input_ch elementos son coordenadas espaciales
#                Últimos input_ch_views elementos son direcciones de vista
               
#         Returns:
#             outputs: Tensor [N, 4] con RGB y densidad
#         """
#         input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
#         h = input_pts
        
#         # Procesar coordenadas espaciales a través de las capas
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts, h], -1)
        
#         # Predecir densidad
#         density = self.density_linear(h)
        
#         # Obtener características para el color
#         feature = self.feature_linear(h)
        
#         if self.use_viewdirs:
#             # Procesar direcciones de vista
#             h = torch.cat([feature, input_views], -1)
            
#             for i, l in enumerate(self.views_linears):
#                 h = self.views_linears[i](h)
#                 h = F.relu(h)
            
#             rgb = self.rgb_linear(h)
#         else:
#             rgb = self.rgb_linear(feature)
        
#         outputs = torch.cat([rgb, density], -1)
#         return outputs


# def positional_encoding(x, L):
#     """
#     Aplica codificación posicional a las coordenadas de entrada.
    
#     Args:
#         x: Tensor de coordenadas [N, 3]
#         L: Número de frecuencias para la codificación
        
#     Returns:
#         encoded: Tensor codificado [N, 3 + 3*2*L]
#     """
#     encoded = [x]
#     for i in range(L):
#         for fn in [torch.sin, torch.cos]:
#             encoded.append(fn(2.**i * np.pi * x))
#     return torch.cat(encoded, -1)


# def get_rays(H, W, focal, c2w):
#     """
#     Genera rayos de cámara para una pose dada.
    
#     Args:
#         H: Altura de la imagen
#         W: Ancho de la imagen
#         focal: Distancia focal
#         c2w: Matriz de transformación cámara-a-mundo [3, 4]
        
#     Returns:
#         rays_o: Orígenes de los rayos [H, W, 3]
#         rays_d: Direcciones de los rayos [H, W, 3]
#     """
#     device = c2w.device
#     i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
#     i = i.t()
#     j = j.t()
    
#     # Coordenadas de pantalla a coordenadas de cámara
#     dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    
#     # Transformar direcciones de cámara a mundo
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    
#     # Origen del rayo (posición de la cámara)
#     rays_o = c2w[:3, -1].expand(rays_d.shape)
    
#     return rays_o, rays_d


# def sample_pdf(bins, weights, N_samples, det=False):
#     """
#     Muestreo por importancia basado en los pesos.
    
#     Args:
#         bins: Bins de distancia [N_rays, N_samples]
#         weights: Pesos para cada bin [N_rays, N_samples]
#         N_samples: Número de muestras a generar
#         det: Si usar muestreo determinístico
        
#     Returns:
#         samples: Muestras de distancia [N_rays, N_samples]
#     """
#     # Obtener PDF
#     weights = weights + 1e-5  # Prevenir nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)
#     cdf = torch.cumsum(pdf, -1)
#     cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    
#     # Tomar muestras uniformes
#     if det:
#         u = torch.linspace(0., 1., steps=N_samples, device=weights.device)
#         u = u.expand(list(cdf.shape[:-1]) + [N_samples])
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=weights.device)
    
#     # Inversión de CDF
#     u = u.contiguous()
#     inds = torch.searchsorted(cdf, u, right=True)
#     below = torch.max(torch.zeros_like(inds-1), inds-1)
#     above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)
    
#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#     bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
#     denom = (cdf_g[..., 1] - cdf_g[..., 0])
#     denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
#     t = (u - cdf_g[..., 0]) / denom
#     samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
#     return samples


# def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0):
#     """
#     Convierte predicciones raw del modelo a RGB y profundidad.
    
#     Args:
#         raw: Predicciones raw del modelo [N_rays, N_samples, 4]
#         z_vals: Valores de profundidad [N_rays, N_samples]
#         rays_d: Direcciones de rayos [N_rays, 3]
#         raw_noise_std: Desviación estándar del ruido
        
#     Returns:
#         rgb_map: Mapa RGB [N_rays, 3]
#         disp_map: Mapa de disparidad [N_rays]
#         acc_map: Mapa de acumulación [N_rays]
#         weights: Pesos [N_rays, N_samples]
#         depth_map: Mapa de profundidad [N_rays]
#     """
#     def raw2alpha(raw, dists, act_fn=F.relu):
#         return 1.0 - torch.exp(-act_fn(raw) * dists)
    
#     # Calcular distancias entre muestras
#     dists = z_vals[..., 1:] - z_vals[..., :-1]
#     dists = torch.cat([dists, torch.tensor([1e10], device=z_vals.device).expand(dists[..., :1].shape)], -1)
#     dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
#     # Extraer RGB y densidad
#     rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
#     noise = 0.
#     if raw_noise_std > 0.:
#         noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std
    
#     # Calcular alpha y pesos
#     alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
#     weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
#     # Renderizar RGB
#     rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    
#     # Calcular profundidad y disparidad
#     depth_map = torch.sum(weights * z_vals, -1)
#     disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
#     acc_map = torch.sum(weights, -1)
    
#     return rgb_map, disp_map, acc_map, weights, depth_map
