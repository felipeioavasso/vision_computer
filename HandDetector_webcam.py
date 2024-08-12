import cv2
from cvzone.HandTrackingModule import HandDetector

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Inicializar o detector de mãos
detector = HandDetector(maxHands=2)

while True:
    # Ler o quadro da câmera
    success, img = cap.read()

    # Detectar mãos
    hands, img = detector.findHands(img)

    # Exibir a imagem
    cv2.imshow("Image", img)

    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
