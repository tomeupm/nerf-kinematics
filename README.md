# Reconstrucción 3D con Neural Radiance Fields (NeRF)

Este proyecto forma parte de un Trabajo de Fin de Máster (TFM) que implementa un sistema de reconstrucción 3D utilizando Neural Radiance Fields (NeRF) a partir de imágenes y datos de posición.

## Descripción

La herramienta permite generar modelos 3D desde conjuntos de imágenes, utilizando diferentes implementaciones de Neural Radiance Fields:

- **NeRF tradicional**: Implementación estándar que funciona en CPU o GPU.
- **instant-ngp**: Implementación acelerada que requiere CUDA, ofreciendo un rendimiento significativamente más rápido.

El sistema es capaz de procesar imágenes con o sin datos de posición de cámara. En caso de no disponer de datos de posición, estos se estiman automáticamente mediante técnicas de Structure from Motion (SfM).

## Estructura del Proyecto

- `main.py`: Punto de entrada principal. Contiene el parser de argumentos y la lógica de control de flujo.
- `preprocessing.py`: Módulo para el procesamiento de imágenes y datos de posición. Implementa las funciones para cargar, validar y preparar los datos para la reconstrucción 3D.
- `pruebas_iniciales.ipynb`: Notebook Jupyter con pruebas y experimentos iniciales del proyecto.

## Requisitos

- Python 3.8 o superior
- Bibliotecas: NumPy, OpenCV, PyTorch (ver `requirements.txt` para una lista completa)
- CUDA (opcional, para usar instant-ngp)
- COLMAP/GLOMAP (para estimación de posiciones de cámara)

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
python main.py --img ruta/a/imagenes [--pos ruta/a/posiciones] [--use_cuda]
```

### Parámetros

- `--img`: (Requerido) Ruta al directorio que contiene las imágenes.
- `--pos`: (Opcional) Ruta al directorio con los archivos JSON de posición de cámara. Si no se proporciona, se estimarán automáticamente.
- `--use_cuda`: (Opcional) Flag para utilizar la implementación acelerada instant-ngp con CUDA.

### Ejemplos

1. Procesar imágenes con posiciones conocidas:
   ```bash
   python main.py --img ./imagenes --pos ./posiciones
   ```

2. Procesar imágenes utilizando instant-ngp (requiere CUDA):
   ```bash
   python main.py --img ./imagenes --pos ./posiciones --use_cuda
   ```

3. Procesar imágenes sin datos de posición (se estimarán automáticamente):
   ```bash
   python main.py --img ./imagenes
   ```

## Formato de Datos

### Imágenes

Las imágenes deben estar en un formato estándar (JPG, PNG) y deben estar todas en el mismo directorio.

### Datos de Posición

Los datos de posición deben estar en archivos JSON con el mismo nombre base que las imágenes correspondientes. Por ejemplo:

- `imagen_001.jpg` → `imagen_001.json`
- `imagen_002.jpg` → `imagen_002.json`

Cada archivo JSON debe contener la información de posición y orientación de la cámara.

## Resultados

Los resultados de la reconstrucción 3D se guardarán en un directorio de salida, que incluirá:

- Modelo 3D reconstruido
- Métricas de calidad
- Visualizaciones desde diferentes ángulos

## Desarrollo Futuro

- Implementación completa de la reconstrucción 3D con NeRF puro
- Integración con instant-ngp
- Mejoras en la estimación de posiciones
- Interfaz gráfica para facilitar el uso

## Contacto

[Tu nombre] - [Tu correo electrónico]

---

Proyecto desarrollado para el Trabajo de Fin de Máster en [Nombre del Máster], Universidad de Navarra.