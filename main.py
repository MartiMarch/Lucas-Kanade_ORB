"""
    Aplicación del algoritmo de Lucas-Kanade. Gracias al flujo óptico se puede detectar el movimiento.
    Combinado con ORB es posible desplazar la cámar evitando falsos positivos.
"""
import cv2
import numpy as np

# Funcion utilizada para dividir en cuatro la imagen
def procesarEsquinasInicialesSubimagen(imagen, x1, x2, y1, y2):
    subimagen = imagen[x1:x2, y1:y2]
    esquinasIniciales = orb.detect(subimagen, mask=None)
    puntos = []
    puntos.append([])
    for n in esquinasIniciales:
        puntos[0].append([np.float32(n.pt)])
    esquinasIniciales = puntos[0]
    esquinasIniciales = np.asarray(esquinasIniciales)
    return esquinasIniciales

# Funcion usada para crear subimagenes
def obtenerSubimagen(imagen, x1, x2, y1, y2):
    return imagen[x1:x2, y1:y2]

# Se dibuja el flujo óptico
def dibujarFlujoOptico(buenasNuevasEsquinas, buenasEsquinasIniciales, x1, x2, y1, y2, imagen, primeraImagen):
    for i, (new, old) in enumerate(zip(buenasNuevasEsquinas, buenasEsquinasIniciales)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(np.zeros_like(primeraImagen[x1:x2, y1:y2]), (a, b), (c, d), color[i].tolist(), 2)
        return cv2.circle(imagen[x1:x2, y1:y2], (a, b), 5, color[i].tolist(), -1), mask

# Se abre la cámara
camara = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Se obtiene el primer frame
_, primeraImagen = camara.read()

# Se transforma a gris
primeraImagenGris = cv2.cvtColor(primeraImagen, cv2.COLOR_BGR2GRAY)

# Se crea un orb
orb = cv2.ORB_create(100)

# Se divide la primera imagen en cuatro
esquinasIniciales1 = procesarEsquinasInicialesSubimagen(primeraImagenGris, 0, int(primeraImagenGris.shape[0]/2), 0, int(primeraImagenGris.shape[1]/2))
esquinasIniciales2 = procesarEsquinasInicialesSubimagen(primeraImagenGris, 0, int(primeraImagenGris.shape[0]/2), int(primeraImagenGris.shape[1]/2), int(primeraImagenGris.shape[1]))
esquinasIniciales3 = procesarEsquinasInicialesSubimagen(primeraImagenGris, int(primeraImagenGris.shape[0]/2), int(primeraImagenGris.shape[0]), 0, int(primeraImagenGris.shape[1]/2))
esquinasIniciales4 = procesarEsquinasInicialesSubimagen(primeraImagenGris, int(primeraImagenGris.shape[0]/2), int(primeraImagenGris.shape[0]), int(primeraImagenGris.shape[1]/2), int(primeraImagenGris.shape[1]))

# Se trata de un vector que representa un color aleatorio
color = np.random.randint(0, 255, (100, 3))

while True:
    # Se van leyendo imágenes
    _, imagen = camara.read()

    # Convertimos el frame a gris
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    """
        (https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)
        Se calcula el flujo óptico. Los parámetros de entrada son los siguientes:

        - 1: la imagen sobre la que comparar el cambio de intensidad. Se constuye una pirámide con ella.

        - 2: imagen en la que el tiempo ha variado.

        - 3: puntos claves iniciales usados como referencia.

        - 4: No se si se refiere a 'err' o 'status'. En el constructor no queda claro.

        - 5: tamaño de la ventna utilizada. Se corresponde con los píxeles vecinos del punto clave.

        - 6: es el nivel de la pirámide. Cuantos más niveles más pequeña será la imágen sobre la que se realizará el cálculo del flujo óptico.

        -7: Criterio de parada (es el mismo que en MeanShift). La ventana dejará de reubicarse o bien cuando se supere cierto número de iteraciones o cuando
            se da cierto criterio de convergencia.

        Parámetros de salida:

        - 1: Siguientes puntos a los que se desplazarán las ventanas.

        - 2: vector 'status' cuyo valores son booleanos. Si se detecta una variación en el flujo óptico de cada uno de los puntos clave se marca como 0. En caso contrario
             vale 1.

        - 3: Cada punto clave tiene un margen de error cuando se calcula el flujo óptico.
    """
    nuevasEsquinas1, estatus1, errores1 = cv2.calcOpticalFlowPyrLK(primeraImagenGris[
                                                                    0:int(primeraImagenGris.shape[0]/2),
                                                                    0:int(primeraImagenGris.shape[1]/2)
                                                                   ],
                                                                   imagenGris[
                                                                    0:int(imagenGris.shape[0]/2),
                                                                    0:int(imagenGris.shape[1]/2)
                                                                   ],
                                                                   esquinasIniciales1, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    nuevasEsquinas2, estatus2, errores2 = cv2.calcOpticalFlowPyrLK(primeraImagenGris[
                                                                    0:int(primeraImagenGris.shape[0]/2),
                                                                    int(primeraImagenGris.shape[1]/2):int(primeraImagenGris.shape[1])
                                                                   ],
                                                                   imagenGris[
                                                                    0:int(imagenGris.shape[0]/2),
                                                                    int(imagenGris.shape[1]/2):int(imagenGris.shape[1])
                                                                   ],
                                                                   esquinasIniciales2, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    nuevasEsquinas3, estatus3, errores3 = cv2.calcOpticalFlowPyrLK(primeraImagenGris[
                                                                    int(primeraImagenGris.shape[0]/2):int(primeraImagenGris.shape[0]),
                                                                    0:int(primeraImagenGris.shape[1]/2)
                                                                   ],
                                                                   imagenGris[
                                                                    int(imagenGris.shape[0]/2):int(imagenGris.shape[0]),
                                                                    0:int(imagenGris.shape[1]/2)
                                                                    ],
                                                                   esquinasIniciales3, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    nuevasEsquinas4, estatus4, errores4 = cv2.calcOpticalFlowPyrLK(primeraImagenGris[
                                                                    int(primeraImagenGris.shape[0]/2):int(primeraImagenGris.shape[0]),
                                                                    int(primeraImagenGris.shape[1]/2):int(primeraImagenGris.shape[1])
                                                                    ],
                                                                   imagenGris[
                                                                    int(imagenGris.shape[0]/2):int(imagenGris.shape[0]),
                                                                    int(imagenGris.shape[1]/2):int(imagenGris.shape[1])
                                                                    ],
                                                                   esquinasIniciales4, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Se seleccionan los mejores puntos clave de las esquinas antiguas mediante el vector 'status'
    buenasEsquinasIniciales1 = esquinasIniciales1[estatus1 == 1]
    buenasEsquinasIniciales2 = esquinasIniciales2[estatus2 == 1]
    buenasEsquinasIniciales3 = esquinasIniciales3[estatus3 == 1]
    buenasEsquinasIniciales4 = esquinasIniciales4[estatus4 == 1]

    # Se seleccionan los mejores puntos clave de las nuevas esquinas calculadas mediate el vector 'status'
    buenasNuevasEsquinas1 = nuevasEsquinas1[estatus1 == 1]
    buenasNuevasEsquinas2 = nuevasEsquinas2[estatus2 == 1]
    buenasNuevasEsquinas3 = nuevasEsquinas3[estatus3 == 1]
    buenasNuevasEsquinas4 = nuevasEsquinas4[estatus4 == 1]

    # Se dibuja el recorrido del flujo óptico
    imagenDibujada1, mask1 = dibujarFlujoOptico(buenasNuevasEsquinas1, buenasEsquinasIniciales1,
                                                0,
                                                int(primeraImagenGris.shape[0]/2),
                                                0,
                                                int(primeraImagenGris.shape[1]/2),
                                                imagen, primeraImagen)
    imagenDibujada2, mask2 = dibujarFlujoOptico(buenasNuevasEsquinas2, buenasEsquinasIniciales2,
                                                0,
                                                int(primeraImagenGris.shape[0]/2),
                                                int(primeraImagenGris.shape[1]/2),
                                                int(primeraImagenGris.shape[1]),
                                                imagen, primeraImagen)
    imagenDibujada3, mask3 = dibujarFlujoOptico(buenasNuevasEsquinas3, buenasEsquinasIniciales3,
                                                int(primeraImagenGris.shape[0]/2),
                                                int(primeraImagenGris.shape[0]),
                                                0,
                                                int(imagenGris.shape[1]/2),
                                                imagen, primeraImagen)
    imagenDibujada4, mask4 = dibujarFlujoOptico(buenasNuevasEsquinas4, buenasEsquinasIniciales4,
                                                int(primeraImagenGris.shape[0]/2),
                                                int(primeraImagenGris.shape[0]),
                                                int(primeraImagenGris.shape[1]/2),
                                                int(primeraImagenGris.shape[1]),
                                                imagen, primeraImagen)

    # Se aplica la máscara sobre la imagen
    imagenSalida1 = cv2.add(imagenDibujada1, mask1)
    imagenSalida2 = cv2.add(imagenDibujada2, mask2)
    imagenSalida3 = cv2.add(imagenDibujada3, mask3)
    imagenSalida4 = cv2.add(imagenDibujada4, mask4)

    # Se muestra la imagen
    cv2.imshow("Lucas-Kanade 1", imagenSalida1)
    cv2.imshow("Lucas-Kanade 2", imagenSalida2)
    cv2.imshow("Lucas-Kanade 3", imagenSalida3)
    cv2.imshow("Lucas-Kanade 4", imagenSalida4)

    if 0 in estatus1:
        print("Movimiento")
    if 0 in estatus2:
        print("Movimiento")
    if 0 in estatus3:
        print("Movimiento")
    if 0 in estatus4:
        print("Movimiento")

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break