import cv2

# Carregar Script/Algoritmo
caminho_classificador = ''
carregar_script = cv2.CascadeClassifier(caminho_classificador)

# Verificar se o classificador foi carregado corretamente
if carregar_script.empty():
    print("Erro ao carregar o classificador Haar. Verifique o caminho e o arquivo.")
    exit()

# Inicializar a captura de imagem
caminho_imagem = ''  # Certifique-se de que a extensão está correta
imagem = cv2.imread(caminho_imagem)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem. Verifique o caminho e o arquivo.")
    exit()

# Converter a imagem para escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar faces
faces = carregar_script.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Imprimir coordenadas das faces detectadas
print(faces)

# Desenhar retângulos ao redor das faces detectadas
for (x, y, l, a) in faces:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

# Definir o nome da janela
window_name = "Faces"

# Criar a janela com uma dimensão inicial
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Redimensionar a janela (largura, altura)
cv2.resizeWindow(window_name, 450, 450)

# Exibir a imagem com as faces detectadas
cv2.imshow(window_name, imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
