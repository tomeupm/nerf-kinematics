import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    """
    Implementación del modelo Neural Radiance Field (NeRF).
    
    Red neuronal que mapea coordenadas 3D (x, y, z) y direcciones de vista (theta, phi)
    a densidad volumétrica y color RGB.
    """
    
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """
        Args:
            D: Número de capas en la red
            W: Número de neuronas por capa
            input_ch: Canales de entrada para coordenadas espaciales
            input_ch_views: Canales de entrada para direcciones de vista
            output_ch: Canales de salida (RGB + densidad)
            skips: Lista de capas donde añadir conexiones residuales
            use_viewdirs: Si usar direcciones de vista para predicción de color
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # Capas para procesamiento de coordenadas espaciales
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        
        # Capa para predecir densidad
        self.density_linear = nn.Linear(W, 1)
        
        # Capa de características antes del color
        self.feature_linear = nn.Linear(W, W)
        
        # Capas para procesamiento de direcciones de vista
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.rgb_linear = nn.Linear(W, 3)
            
    def forward(self, x):
        """
        Forward pass del modelo NeRF.
        
        Args:
            x: Tensor de entrada [N, input_ch + input_ch_views]
               Primeros input_ch elementos son coordenadas espaciales
               Últimos input_ch_views elementos son direcciones de vista
               
        Returns:
            outputs: Tensor [N, 4] con RGB y densidad
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        
        # Procesar coordenadas espaciales a través de las capas
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        # Predecir densidad
        density = self.density_linear(h)
        
        # Obtener características para el color
        feature = self.feature_linear(h)
        
        if self.use_viewdirs:
            # Procesar direcciones de vista
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
        else:
            rgb = self.rgb_linear(feature)
        
        outputs = torch.cat([rgb, density], -1)
        return outputs


def positional_encoding(x, L):
    """
    Aplica codificación posicional a las coordenadas de entrada.
    
    Args:
        x: Tensor de coordenadas [N, 3]
        L: Número de frecuencias para la codificación
        
    Returns:
        encoded: Tensor codificado [N, 3 + 3*2*L]
    """
    encoded = [x]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            encoded.append(fn(2.**i * np.pi * x))
    return torch.cat(encoded, -1)


def get_rays(H, W, focal, c2w):
    """
    Genera rayos de cámara para una pose dada.
    
    Args:
        H: Altura de la imagen
        W: Ancho de la imagen
        focal: Distancia focal
        c2w: Matriz de transformación cámara-a-mundo [3, 4]
        
    Returns:
        rays_o: Orígenes de los rayos [H, W, 3]
        rays_d: Direcciones de los rayos [H, W, 3]
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()
    j = j.t()
    
    # Coordenadas de pantalla a coordenadas de cámara
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    
    # Transformar direcciones de cámara a mundo
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    
    # Origen del rayo (posición de la cámara)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False):
    """
    Muestreo por importancia basado en los pesos.
    
    Args:
        bins: Bins de distancia [N_rays, N_samples]
        weights: Pesos para cada bin [N_rays, N_samples]
        N_samples: Número de muestras a generar
        det: Si usar muestreo determinístico
        
    Returns:
        samples: Muestras de distancia [N_rays, N_samples]
    """
    # Obtener PDF
    weights = weights + 1e-5  # Prevenir nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    
    # Tomar muestras uniformes
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    
    # Inversión de CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0):
    """
    Convierte predicciones raw del modelo a RGB y profundidad.
    
    Args:
        raw: Predicciones raw del modelo [N_rays, N_samples, 4]
        z_vals: Valores de profundidad [N_rays, N_samples]
        rays_d: Direcciones de rayos [N_rays, 3]
        raw_noise_std: Desviación estándar del ruido
        
    Returns:
        rgb_map: Mapa RGB [N_rays, 3]
        disp_map: Mapa de disparidad [N_rays]
        acc_map: Mapa de acumulación [N_rays]
        weights: Pesos [N_rays, N_samples]
        depth_map: Mapa de profundidad [N_rays]
    """
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - torch.exp(-act_fn(raw) * dists)
    
    # Calcular distancias entre muestras
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Extraer RGB y densidad
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
    
    # Calcular alpha y pesos
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    # Renderizar RGB
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    
    # Calcular profundidad y disparidad
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    
    return rgb_map, disp_map, acc_map, weights, depth_map
