"""
    Aplicación del algoritmo de Lucas-Kanade. Gracias al flujo óptico se puede detectar el movimiento.
    Combinado con ORB es posible desplazar la cámar evitando falsos positivos.
"""
import cv2
import numpy as np

# Se abre la cámara
camara = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Se obtiene el primer frame
_, primeraImagen = camara.read()

# Se transforma a gris
primeraImagenGris = cv2.cvtColor(primeraImagen, cv2.COLOR_BGR2GRAY)

# Se crea un orb
orb = cv2.ORB_create(100)

# Puntos clave obtenidos de la primera imagen
esquinasIniciales = orb.detect(primeraImagenGris, mask=None)
puntos = []
puntos.append([])
for n in esquinasIniciales:
    puntos[0].append([np.float32(n.pt)])
esquinasIniciales = puntos[0]
esquinasIniciales = np.asarray(esquinasIniciales)

# Se crea una máscara de la primera imagen
mascara = np.zeros_like(primeraImagenGris)

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
    nuevasEsquinas, estatus, errores = cv2.calcOpticalFlowPyrLK(primeraImagenGris, imagenGris, esquinasIniciales, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Se seleccionan los mejores puntos clave de las esquinas antiguas mediante el vector 'status'
    buenasEsquinasIniciales = esquinasIniciales[estatus == 1]

    # Se seleccionan los mejores puntos clave de las nuevas esquinas calculadas mediate el vector 'status'
    buenasNuevasEsquinas = nuevasEsquinas[estatus == 1]

    # Se dibuja el recorrido del flujo óptico
    for i, (new, old) in enumerate(zip(buenasNuevasEsquinas, buenasEsquinasIniciales)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(np.zeros_like(primeraImagen), (a, b), (c, d), color[i].tolist(), 2)
        imagen = cv2.circle(imagen, (a, b), 5, color[i].tolist(), -1)

    # Se aplica la máscara sobre la imagen
    imagenSalida = cv2.add(imagen, mask)

    # Se muestra la imagen
    cv2.imshow("Lucas-Kanade", imagenSalida)

    if 0 in estatus:
        print("Movimiento")

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break