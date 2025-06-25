import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_loader import load_nerf_data
from nerf_model import create_nerf, get_rays, get_rays_np
# from train_nerf import train_nerf, TrainingArgs
from render import render#, render_novel_views, extract_mesh

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))

def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)



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
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='Número de neuronas por capa en la red NeRF')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='Número de capas en la red NeRF')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # DE MOMENTO SIN ESTOS PARAMETROS WTF NO SE QUE SON
    # parser.add_argument('--N_samples', type=int, default=64,
    #                     help='Número de muestras por rayo (coarse)')
    # parser.add_argument('--N_importance', type=int, default=128,
    #                     help='Número de muestras adicionales por rayo (fine)')
    
    
    # rendering options
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    
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
    images, poses, render_poses, hwf, i_split = load_nerf_data(args.img, args.pos, half_res=args.half_res)
    
    # Extraer información de los datos cargados
    H, W, focal = hwf
    i_train, i_val, i_test = i_split
    
    print(f"Datos cargados exitosamente:")
    print(f"  - Imágenes: {images.shape}")
    print(f"  - Poses: {poses.shape}")
    print(f"  - Resolución: {H}x{W}, focal: {focal:.2f}")
    print(f"  - Train/Val/Test: {len(i_train)}/{len(i_val)}/{len(i_test)}")
    print(f"  - Render poses: {len(render_poses) if render_poses is not None else 0}")

    # IN-PROGRESS: enganchar siguiendo nerf original

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    
    near = 2.
    far = 6.
    
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)
    
    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
     # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        
        # Asegurar que las imágenes solo tengan 3 canales (RGB)
        if images.shape[-1] == 4:
            images = images[..., :3]  # Remover canal alpha si existe
        
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    N_iters = args.N_iters  # Usar el valor del argumento
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    
    # Summary writers
    import tensorflow.compat.v1 as tf_v1
    tf_v1.disable_v2_behavior()
    writer = tf_v1.summary.FileWriter(os.path.join(args.basedir, 'summaries'))

    for i in tqdm(range(N_iters), desc="Entrenando NeRF"):

        # Sample random ray batch

        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        #####           end            #####
        
        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)
                
        # Logging
        if i % getattr(args, 'i_print', 100) == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")
                
        #TODO: Implementar el resto de logging y guardado de checkpoints

    
    
    
    # ----------------------------------------------------------
    
    # # Crear configuración de entrenamiento
    # train_args = TrainingArgs()
    
    # # Actualizar configuración con argumentos de línea de comandos
    # for key, value in vars(args).items():
    #     if hasattr(train_args, key):
    #         setattr(train_args, key, value)
    
    # # Ejecutar acciones solicitadas
    # if args.train:
    #     print("\n" + "="*50)
    #     print("ENTRENANDO MODELO NERF")
    #     print("="*50)
    #     train_nerf(train_args, imgs, poses, hwf, i_split, render_poses)
    
    # if args.render:
    #     print("\n" + "="*50)
    #     print("RENDERIZANDO VISTAS NOVELES")
    #     print("="*50)
    #     if render_poses is not None:
    #         render_novel_views(train_args, imgs, poses, hwf, i_split, render_poses)
    #     else:
    #         print("Error: No hay poses de renderizado disponibles.")
    
    # if args.extract_mesh:
    #     print("\n" + "="*50)
    #     print("EXTRAYENDO MALLA 3D")
    #     print("="*50)
    #     extract_mesh(train_args, imgs, poses, hwf)
    
    # print("\n" + "="*50)
    # print("PROCESO COMPLETADO")
    # print("="*50)


# Ejecutar el programa cuando se llame como script principal
if __name__ == '__main__':
    main()
