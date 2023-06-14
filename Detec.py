# Se importan librerías
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

img = cv2.imread('img28.jpg')
height, width, channels = img.shape

print("Dimensiones de la imagen:")
print("Ancho:", width)
print("Altura:", height)

imagen = cv2.resize(img, (404, 303), interpolation = cv2.INTER_AREA)  # Cambiar tamaño de la imagen
imagen.shape

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
cv2_imshow(imagen_gris)

def detectar_bordes_sobel(imagen_gris):
    
    # Aplicar las derivadas de Sobel en los ejes x e y
    gradiente_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcular la magnitud del gradiente
    magnitud_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)
    
    # Aplicar umbral para obtener los bordes
    umbral = np.max(magnitud_gradiente) * 0.1
    bordes = np.uint8(magnitud_gradiente > umbral) * 255
    
    # Contar la cantidad de píxeles de borde en toda la imagen
    cantidad_bordes_total = np.sum(bordes) // 255
    
    # Obtener la región de interés desde el 55% de la imagen hacia arriba
    altura_roi = int(bordes.shape[0] * 0.55)
    roi = bordes[:altura_roi, :]
    
    # Contar la cantidad de píxeles de borde en la región de interés
    cantidad_bordes_roi = np.sum(roi) // 255
    
    return bordes, cantidad_bordes_total, cantidad_bordes_roi

# Detectar los bordes utilizando las derivadas de Sobel
bordes, cantidad_bordes_total, cantidad_bordes_roi = detectar_bordes_sobel(imagen_gris)

# Mostrar la imagen original y los bordes detectados
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original'), plt.axis('off')
plt.subplot(122), plt.imshow(bordes, cmap='gray')
plt.title('Bordes Detectados (Total: {}, Región de interés: {})'.format(cantidad_bordes_total, cantidad_bordes_roi))
plt.axis('off')
plt.show()

if 7000 <= cantidad_bordes_total <= 7899 and 3050 <= cantidad_bordes_roi <= 3590:
  print('NORMAL')
elif 8295 <= cantidad_bordes_total <= 9050 and 4105 <= cantidad_bordes_roi <= 4760:
  print('GAFAS')
elif 7900 <= cantidad_bordes_total <= 8550 and 3210 <= cantidad_bordes_roi <= 3680:
  print('GORRA')
else:
  print('No se pudo determinar')