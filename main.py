"""
    Aplicación del algoritmo de Lucas-Kanade. Gracias al flujo óptico se puede detectar el movimiento.
    Combinado con ORB es posible desplazar la cámara evitando falsos positivos. Se divide la imagen en cuatro
    imágenes para poder distribuir los puntos clave un poco más uniformemente.
"""
import cv2
import numpy as np
import time

""" procesarEsquinasInicialesSubimagen

    Función utilizada para dividir en cuatro la imagen.

    [In]    imagen: imagen tomada como referencia.
    [In]    x1: punto x inicial sobre el que se empieza la subdivision de la imagen
    [In]    x2: punto x final sobre el que se empieza la subdivision de la imagen
    [In]    y1: punto y inicial sobre el que se empieza la subdivision de la imagen
    [In]    y2: punto y final sobre el que se empieza la subdivision de la imagen
    [Out]   Puntos clave que se utilizaran para comparar los puntos clave actuales.
    [Out]   Descriptores de los puntos iniciales.
"""
def procesarEsquinasInicialesSubimagen(subimagen):

    # Se obtienen los puntos claves
    esquinasIniciales = orb.detect(subimagen, mask=None)

    # Las esquinas iniciales de adaptan al formato correcto
    puntos = []
    puntos.append([])
    for n in esquinasIniciales:
        puntos[0].append([np.float32(n.pt)])
    esquinasIniciales = puntos[0]
    esquinasIniciales = np.asarray(esquinasIniciales)

    return esquinasIniciales

""" dibujarFlujoOptico
    
    Función utilizada para dibujar el flujo óptico.
    
    [In]    buenasNuevasEsquinas: son los nuevos puntos claves actualizados.
    [In]    buenasEsquinasIniciales: son los puntos clave iniciales.
    [In]    x1: punto x inicial sobre el que se empieza la subdivision de la imagen
    [In]    x2: punto x final sobre el que se empieza la subdivision de la imagen
    [In]    y1: punto y inicial sobre el que se empieza la subdivision de la imagen
    [In]    y2: punto y final sobre el que se empieza la subdivision de la imagen
    [In]    imagen: imagen que se ha consumido de la cámara
    [In]    primeraImagen: imagen de referencia sobre la que se comparará la imagen consumida
    [In]    es necesario pasarle la subimagen resultante conn los circulos dibujadaos para evitar problemas con el ámbito de la varaible
    [In]    es necesario pasarle la máscara resultante conn los circulos dibujadaos para evitar problemas con el ámbito de la varaible
    [Out]   Subimagen de la imagen original con el flujo óptico dibujado
    [Out]   Máscara que hay que aplicar sobre la subimagen
"""
def dibujarFlujoOptico(buenasNuevasEsquinas, buenasEsquinasIniciales, imagen, subimagenDibujada, mask):

    for i, (new, old) in enumerate(zip(buenasNuevasEsquinas, buenasEsquinasIniciales)):
        a, b = new.ravel()
        c, d = old.ravel()
        subimagenDibujada = cv2.circle(imagen, (a, b), 5, color[i].tolist(), -1)
        mask = cv2.line(np.zeros_like(imagen), (a, b), (c, d), color[i].tolist(), 2)

    return subimagenDibujada, mask

""" getTime

    Función utilizada para conocer el tiempo actual en milisegundos.
    
    [Out]   Tiempo actual en milisegundos
"""
def getTime():

    return round(time.time() * 1000)

""" obtenerEsquinasValidas
    
    Evita que el programa empiece si tener unas esquinas iniciales para cada una de las subimágenes.
    
    [Out]   Esquinas inicilaes de cada subimagen
    [Out]   Subimagenes tomadas como referencia

"""
def obtenerEsquinasValidas():

    # Lista que contiene las esquinas iniciales
    esquinasIniciales = []

    # Lista que contiene las subimagenes iniciales
    imagenesIniciales = []

    # Se crean los vectores de las esquinas iniciales
    esquinasIniciales1 = []
    esquinasIniciales2 = []
    esquinasIniciales3 = []
    esquinasIniciales4 = []

    while len(esquinasIniciales1) <= 0 or len(esquinasIniciales2) <= 0 or len(esquinasIniciales3) <= 0 or len(esquinasIniciales4) <= 0:

        # Se vacian las listas
        esquinasIniciales = []
        imagenesIniciales = []

        # Se obtiene el primer frame
        _, primeraImagen = camara.read()

        # Se transforma a gris
        primeraImagenGris = cv2.cvtColor(primeraImagen, cv2.COLOR_BGR2GRAY)

        # Se subdivide la primera imagen en gris en cuatro partes
        primeraImagenGris1 = primeraImagenGris[0:int(primeraImagenGris.shape[0] / 2), 0:int(primeraImagenGris.shape[1] / 2)]
        primeraImagenGris2 = primeraImagenGris[0:int(primeraImagenGris.shape[0] / 2), int(primeraImagenGris.shape[1] / 2):int(primeraImagenGris.shape[1])]
        primeraImagenGris3 = primeraImagenGris[int(primeraImagenGris.shape[0] / 2):int(primeraImagenGris.shape[0]), 0:int(primeraImagenGris.shape[1] / 2)]
        primeraImagenGris4 = primeraImagenGris[int(primeraImagenGris.shape[0] / 2):int(primeraImagenGris.shape[0]), int(primeraImagenGris.shape[1] / 2):int(primeraImagenGris.shape[1])]

        # Se divide la primera imagen en cuatro
        esquinasIniciales1 = procesarEsquinasInicialesSubimagen(primeraImagenGris1)
        esquinasIniciales2 = procesarEsquinasInicialesSubimagen(primeraImagenGris2)
        esquinasIniciales3 = procesarEsquinasInicialesSubimagen(primeraImagenGris3)
        esquinasIniciales4 = procesarEsquinasInicialesSubimagen(primeraImagenGris4)

        # Se llena la lista de esquinas
        esquinasIniciales.append(esquinasIniciales1)
        esquinasIniciales.append(esquinasIniciales2)
        esquinasIniciales.append(esquinasIniciales3)
        esquinasIniciales.append(esquinasIniciales4)

        # Se llena la lista de imagenes iniciales
        imagenesIniciales.append(primeraImagenGris1)
        imagenesIniciales.append(primeraImagenGris2)
        imagenesIniciales.append(primeraImagenGris3)
        imagenesIniciales.append(primeraImagenGris4)

    return esquinasIniciales, imagenesIniciales

""" ------------------------------ ***
        EMPIEZA EL PROGRAMA
*** ------------------------------ """

# Se abre la cámara
camara = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Se crea un orb
orb = cv2.ORB_create(100)

# Se trata de un vector que representa un color aleatorio
color = np.random.randint(0, 255, (100, 3))

# Tiempo en el que se tomo la primera imagen de referencia
tiempoInicial = getTime()

# Lista de esquinas iniciales
esquinasIniciales = []

# Lista de subimagenes iniciales
imagenesIniciales = []

# Se obtiene las esqwuinas y subimágenes
esquinasIniciales, imagenesIniciales = obtenerEsquinasValidas()

while True:
    # Se van leyendo imágenes
    _, imagen = camara.read()

    # Se convierte la imagen original en subimagenes
    imagen1 = imagen[0:int(imagen.shape[0]/2), 0:int(imagen.shape[1]/2)]
    imagen2 = imagen[0:int(imagen.shape[0]/2), int(imagen.shape[1]/2):int(imagen.shape[1])]
    imagen3 = imagen[int(imagen.shape[0]/2):int(imagen.shape[0]), 0:int(imagen.shape[1]/2)]
    imagen4 = imagen[int(imagen.shape[0]/2):int(imagen.shape[0]), int(imagen.shape[1]/2):int(imagen.shape[1])]

    # Convertimos el frame a gris
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Se extraen cuatro subimágenes de la imagen gris
    imagenGris1 = imagenGris[0:int(imagen.shape[0]/2), 0:int(imagen.shape[1]/2)]
    imagenGris2 = imagenGris[0:int(imagen.shape[0]/2), int(imagen.shape[1]/2):int(imagen.shape[1])]
    imagenGris3 = imagenGris[int(imagen.shape[0]/2):int(imagen.shape[0]), 0:int(imagen.shape[1]/2)]
    imagenGris4 = imagenGris[int(imagen.shape[0]/2):int(imagen.shape[0]), int(imagen.shape[1]/2):int(imagen.shape[1])]

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
    nuevasEsquinas1, estatus1, errores1 = cv2.calcOpticalFlowPyrLK(imagenesIniciales[0], imagenGris1, esquinasIniciales[0], None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    nuevasEsquinas2, estatus2, errores2 = cv2.calcOpticalFlowPyrLK(imagenesIniciales[1], imagenGris2, esquinasIniciales[1], None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    nuevasEsquinas3, estatus3, errores3 = cv2.calcOpticalFlowPyrLK(imagenesIniciales[2], imagenGris3, esquinasIniciales[2], None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    nuevasEsquinas4, estatus4, errores4 = cv2.calcOpticalFlowPyrLK(imagenesIniciales[3], imagenGris4, esquinasIniciales[3], None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Se seleccionan los mejores puntos clave de las esquinas antiguas mediante el vector 'status'
    buenasEsquinasIniciales1 = esquinasIniciales[0][estatus1 == 1]
    buenasEsquinasIniciales2 = esquinasIniciales[1][estatus2 == 1]
    buenasEsquinasIniciales3 = esquinasIniciales[2][estatus3 == 1]
    buenasEsquinasIniciales4 = esquinasIniciales[3][estatus4 == 1]

    # Se seleccionan los mejores puntos clave de las nuevas esquinas calculadas mediate el vector 'status'
    buenasNuevasEsquinas1 = nuevasEsquinas1[estatus1 == 1]
    buenasNuevasEsquinas2 = nuevasEsquinas2[estatus2 == 1]
    buenasNuevasEsquinas3 = nuevasEsquinas3[estatus3 == 1]
    buenasNuevasEsquinas4 = nuevasEsquinas4[estatus4 == 1]

    # Se dibuja el recorrido del flujo óptico
    temp = 0
    imagenDibujada1, mask1 = dibujarFlujoOptico(buenasNuevasEsquinas1, buenasEsquinasIniciales1, imagen1, temp, temp)
    imagenDibujada2, mask2 = dibujarFlujoOptico(buenasNuevasEsquinas2, buenasEsquinasIniciales2, imagen2, temp, temp)
    imagenDibujada3, mask3 = dibujarFlujoOptico(buenasNuevasEsquinas3, buenasEsquinasIniciales3, imagen3, temp, temp)
    imagenDibujada4, mask4 = dibujarFlujoOptico(buenasNuevasEsquinas4, buenasEsquinasIniciales4, imagen4, temp, temp)

    # Se aplica la máscara sobre la imagen
    imagenSalida1 = cv2.add(imagenDibujada1, mask1)
    imagenSalida2 = cv2.add(imagenDibujada2, mask2)
    imagenSalida3 = cv2.add(imagenDibujada3, mask3)
    imagenSalida4 = cv2.add(imagenDibujada4, mask4)
    
    # Se muestra la imagen
    cv2.imshow("Lucas-Kanade_ORB 1", imagenSalida1)
    cv2.imshow("Lucas-Kanade_ORB 2", imagenSalida2)
    cv2.imshow("Lucas-Kanade_ORB 3", imagenSalida3)
    cv2.imshow("Lucas-Kanade_ORB 4", imagenSalida4)

    # Si algunos de los estatus esta a 0 significa que existe movimiento
    if 0 in estatus1 or 0 in estatus2 or 0 in estatus3 or 0 in estatus4:
        print("Movimiento")

    # Si han pasado 15 minutos los puntos clave se renuevan
    tiempoActual = getTime()
    if tiempoActual - tiempoInicial >= 900000:
    #if tiempoActual - tiempoInicial >= 60000:
        # Se renueva el tiempo
        tiempoInicial = tiempoActual

        # Se obtiene las esqwuinas y subimágenes
        esquinasIniciales, imagenesIniciales = obtenerEsquinasValidas()

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break