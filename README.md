# Reconstrucción 3D con Neural Radiance Fiel```bash
python main.py --img ruta/a/imagenes --pos ruta/al/poses.txt [--use_cuda]
```

### Parámetros

- `--img`: (Requerido) Ruta al directorio que contiene las imágenes.
- `--pos`: (Requerido) Ruta al archivo `poses.txt` que contiene las matrices de transformación 4x4.
- `--use_cuda`: (Opcional) Flag para utilizar la implementación acelerada instant-ngp con CUDA.

Este proyecto forma parte de un Trabajo de Fin de Máster (TFM) que implementa un sistema de reconstrucción 3D utilizando Neural Radiance Fields (NeRF) a partir de imágenes y datos de posición.

## Descripción

La herramienta permite generar modelos 3D desde conjuntos de imágenes, utilizando diferentes implementaciones de Neural Radiance Fields:

- **NeRF tradicional**: Implementación estándar que funciona en CPU o GPU.
- **instant-ngp**: Implementación acelerada que requiere CUDA, ofreciendo un rendimiento significativamente más rápido.

El sistema procesa imágenes junto con datos de posición de cámara en formato específico mediante un archivo `poses.txt` que contiene matrices de transformación 4x4.

## Estructura del Proyecto

- `main.py`: Punto de entrada principal. Contiene el parser de argumentos y la lógica de control de flujo.
- `preprocessing.py`: Módulo para el procesamiento de imágenes y datos de posición. Implementa las funciones para cargar, validar y preparar los datos para la reconstrucción 3D.
- `pruebas_iniciales.ipynb`: Notebook Jupyter con pruebas y experimentos iniciales del proyecto.

## Requisitos

- Python 3.8 o superior
- Bibliotecas: NumPy, OpenCV, PyTorch (ver `requirements.txt` para una lista completa)
- CUDA (opcional, para usar instant-ngp)

## Instalación

1. Clonar este repositorio:
   ```bash
   git clone https://tu-repositorio/tfm.git
   cd tfm
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. (Opcional) Si desea utilizar la implementación acelerada instant-ngp, asegúrese de tener CUDA instalado y configurado correctamente.

## Uso

### Comando Básico

```bash
python main.py --img ruta/a/imagenes [--pos ruta/a/posiciones_o_poses.txt] [--use_cuda]
```

### Parámetros

- `--img`: (Requerido) Ruta al directorio que contiene las imágenes.
- `--pos`: (Opcional) Ruta al directorio con los archivos JSON de posición de cámara o ruta al archivo poses.txt. Si no se proporciona, se estimarán automáticamente.
- `--use_cuda`: (Opcional) Flag para utilizar la implementación acelerada instant-ngp con CUDA.

### Ejemplos

1. Procesar imágenes con archivo poses.txt:
   ```bash
   python main.py --img ./imagenes --pos ./data/poses.txt
   ```

2. Procesar imágenes utilizando instant-ngp (requiere CUDA):
   ```bash
   python main.py --img ./imagenes --pos ./data/poses.txt --use_cuda
   ```

4. Procesar imágenes usando un archivo poses.txt:
   ```bash
   python main.py --img ./data/PoseImage/images_robot --pos ./data/PoseImage/poses.txt --use_cuda
   ```

## Formato de Datos

### Imágenes

Las imágenes deben estar en un formato estándar (JPG, PNG) y deben estar todas en el mismo directorio.

### Datos de Posición

Los datos de posición deben proporcionarse mediante un archivo `poses.txt` que contiene matrices de transformación 4x4 en formato específico.

El archivo `poses.txt` debe contener una matriz de transformación por cada imagen, donde cada matriz se corresponde secuencialmente con las imágenes ordenadas numéricamente (la primera matriz para la imagen que termina en 0, la segunda para la que termina en 1, etc.).

**Formato de las matrices en poses.txt:**
```
[valor11, valor12, valor13, valor14;
 valor21, valor22, valor23, valor24;
 valor31, valor32, valor33, valor34;
 valor41, valor42, valor43, valor44];
```

## Resultados

Los resultados de la reconstrucción 3D se guardarán en un directorio de salida, que incluirá:

- Modelo 3D reconstruido
- Métricas de calidad
- Visualizaciones desde diferentes ángulos

## Desarrollo Futuro

- Implementación completa de la reconstrucción 3D con NeRF puro
- Integración con instant-ngp
- Interfaz gráfica para facilitar el uso

## Contacto

[Tu nombre] - [Tu correo electrónico]

---

Proyecto desarrollado para el Trabajo de Fin de Máster en [Nombre del Máster], Universidad de Navarra.