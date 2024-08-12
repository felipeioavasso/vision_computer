import cv2

# Carregar o classificador em cascata para detecção de faces
carregar_scritp = cv2.CascadeClassifier('haar-cascade-files-master/haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Ler o quadro atual da webcam
    ret, frame = cap.read()

    # Verificar se o quadro foi lido corretamente
    if not ret:
        break

    # Converter o quadro para escala de cinza
    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no quadro
    faces = carregar_scritp.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Exibir o quadro com as detecções de face
    cv2.imshow("Detecção de Faces", frame)

    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas abertas
cap.release()
cv2.destroyAllWindows()
