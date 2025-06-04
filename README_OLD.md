# Reconstrucción 3D con Neural Radiance Fields (NeRF)

Este proyecto forma parte de un Trabajo de Fin de Máster (TFM) que implementa un sistema de reconstrucción 3D utilizando Neural Radiance Fields (NeRF) a partir de imágenes y datos de posición de cámara.

## Descripción

La herramienta permite generar modelos 3D desde conjuntos de imágenes utilizando Neural Radiance Fields. El sistema procesa imágenes junto con datos de posición de cámara en formato `poses.txt` que contiene matrices de transformación 4x4, y ofrece funcionalidades completas de entrenamiento, renderizado y extracción de malla 3D.

## Estructura del Proyecto Modular

El proyecto ha sido reestructurado siguiendo una arquitectura modular para mejorar la mantenibilidad y escalabilidad:

- **`main.py`**: Punto de entrada principal con parser de argumentos y coordinación del pipeline completo
- **`data_loader.py`**: Módulo para carga y preprocesamiento de datos (imágenes y poses de cámara)
- **`nerf_model.py`**: Definición del modelo NeRF con arquitectura de red neuronal y funciones de renderizado volumétrico
- **`train_nerf.py`**: Pipeline de entrenamiento con optimización y gestión de checkpoints
- **`render.py`**: Funcionalidades de renderizado, síntesis de vistas noveles y extracción de malla 3D
- **`requirements.txt`**: Dependencias del proyecto
- **`preprocessing.py`**: Módulo original (mantenido para compatibilidad)
- **`pruebas_iniciales.ipynb`**: Notebook Jupyter con pruebas y experimentos iniciales

## Características Principales

- **Entrenamiento completo de NeRF**: Pipeline de entrenamiento end-to-end con optimización automática
- **Renderizado de vistas noveles**: Generación de nuevas perspectivas de la escena 3D
- **Extracción de malla 3D**: Exportación de geometría 3D usando algoritmo Marching Cubes
- **Configuración flexible**: Parámetros ajustables para diferentes escenarios y calidad
- **Monitoreo de progreso**: Barras de progreso y métricas durante el entrenamiento

## Requisitos

- Python 3.8 o superior
- PyTorch 1.8+ con soporte GPU (recomendado)
- Bibliotecas: NumPy, ImageIO, Matplotlib, tqdm, scikit-image
- CUDA (recomendado para entrenamiento eficiente)

## Instalación

1. Clonar este repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd tfm
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Comandos Principales

**Entrenamiento básico:**
```bash
python main.py --img ruta/a/imagenes --pos ruta/al/poses.txt --train
```

**Renderizado de vistas noveles:**
```bash
python main.py --img ruta/a/imagenes --pos ruta/al/poses.txt --render
```

**Extracción de malla 3D:**
```bash
python main.py --img ruta/a/imagenes --pos ruta/al/poses.txt --extract_mesh
```

**Pipeline completo:**
```bash
python main.py --img ruta/a/imagenes --pos ruta/al/poses.txt --train --render --extract_mesh
```

### Parámetros Principales

- `--img`: (Requerido) Ruta al directorio que contiene las imágenes
- `--pos`: (Requerido) Ruta al archivo `poses.txt` con matrices de transformación 4x4
- `--train`: Entrenar el modelo NeRF
- `--render`: Renderizar vistas noveles
- `--extract_mesh`: Extraer malla 3D del modelo entrenado
- `--half_res`: Reducir resolución a la mitad para acelerar entrenamiento

### Parámetros de Configuración

- `--N_iters`: Número de iteraciones de entrenamiento (por defecto: 200,000)
- `--lrate`: Learning rate (por defecto: 5e-4)
- `--N_samples`: Muestras por rayo (coarse) (por defecto: 64)
- `--N_importance`: Muestras adicionales por rayo (fine) (por defecto: 128)
- `--basedir`: Directorio base para logs y checkpoints (por defecto: ./logs)
- `--expname`: Nombre del experimento (por defecto: nerf_experiment)

### Ejemplos de Uso

1. **Entrenamiento con configuración personalizada:**
   ```bash
   python main.py --img ./data/images --pos ./data/poses.txt --train \
                   --N_iters 100000 --lrate 1e-3 --expname mi_experimento
   ```

2. **Entrenamiento con resolución reducida:**
   ```bash
   python main.py --img ./data/images --pos ./data/poses.txt --train --half_res
   ```

3. **Solo renderizado (requiere modelo entrenado):**
   ```bash
   python main.py --img ./data/images --pos ./data/poses.txt --render \
                   --expname mi_experimento_entrenado
   ```

4. **Ejemplo con datos del proyecto:**
   ```bash
   python main.py --img ./data/PoseImage/images_robot --pos ./data/PoseImage/poses.txt --train
   ```

## Formato de Datos

### Imágenes

Las imágenes deben estar en un formato estándar (JPG, PNG) y ubicadas en un directorio. El sistema las ordenará alfabéticamente para asociarlas con las poses correspondientes.

### Archivo poses.txt

El archivo `poses.txt` contiene matrices de transformación 4x4 que representan la posición y orientación de la cámara para cada imagen. El formato debe ser:

```
[r11, r12, r13, t1;
 r21, r22, r23, t2;
 r31, r32, r33, t3;
 0,   0,   0,   1];
```

Donde:
- `r11-r33`: Matriz de rotación 3x3
- `t1-t3`: Vector de translación
- Cada matriz corresponde secuencialmente a las imágenes ordenadas alfabéticamente

**Ejemplo:**
```
[0.9999, 0.0087, -0.0122, 0.1234;
 -0.0087, 0.9999, 0.0015, 0.5678;
 0.0122, -0.0014, 0.9999, 0.9012;
 0, 0, 0, 1];
[0.9998, 0.0175, -0.0087, 0.1345;
 -0.0175, 0.9998, 0.0035, 0.5789;
 0.0087, -0.0034, 0.9999, 0.9123;
 0, 0, 0, 1];
```

## Pipeline de Entrenamiento

### Fases del Entrenamiento

1. **Carga de Datos**: Procesamiento de imágenes y poses de cámara
2. **Inicialización del Modelo**: Creación de redes coarse y fine
3. **Entrenamiento**: Optimización usando muestreo de rayos y renderizado volumétrico
4. **Guardado de Checkpoints**: Almacenamiento periódico del estado del modelo

### Monitoreo

Durante el entrenamiento se muestran:
- Pérdida (loss) actual
- Progreso de iteraciones
- Tiempo estimado restante
- PSNR (Peak Signal-to-Noise Ratio) en conjunto de validación

## Resultados

Los resultados se guardan en el directorio especificado por `--basedir/--expname`:

- **Checkpoints**: Estados del modelo para continuar entrenamiento
- **Imágenes renderizadas**: Vistas noveles generadas
- **Malla 3D**: Archivo `.ply` con la geometría extraída
- **Videos**: Animaciones de vistas orbitales (si se especifica)

## Resolución de Problemas

### Errores Comunes

1. **Error de memoria GPU**: Usar `--half_res` o reducir `N_samples`
2. **Poses incorrectas**: Verificar formato del archivo `poses.txt`
3. **Imágenes no encontradas**: Comprobar la ruta del directorio de imágenes

### Optimización

- **GPU**: Usar CUDA si está disponible (detectado automáticamente)
- **Memoria**: Ajustar tamaño de batch implícitamente via `N_samples`
- **Velocidad**: Usar `--half_res` para pruebas rápidas

## Desarrollo Futuro

- Integración con técnicas de estimación de poses automática
- Soporte para diferentes formatos de entrada
- Optimizaciones para entrenamiento más rápido
- Interfaz gráfica para facilitar el uso

## Contacto

Proyecto desarrollado para el Trabajo de Fin de Máster, Universidad de Navarra.

## Licencia

Este proyecto está desarrollado con fines académicos como parte de un TFM.
