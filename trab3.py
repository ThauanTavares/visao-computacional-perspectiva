import cv2
import numpy as np

pontos = []

def clique(evento, x, y, flags, param):
    global pontos
    if evento == cv2.EVENT_LBUTTONDOWN:
        pontos.append([x, y])
        print(f"Ponto selecionado: ({x}, {y})")

imagem = cv2.imread("download.jpeg")
imagem_copia = imagem.copy()

cv2.imshow("Selecione 4 pontos (TL, TR, BR, BL)", imagem)
cv2.setMouseCallback("Selecione 4 pontos (TL, TR, BR, BL)", clique)

print("Clique nos 4 cantos da imagem na ordem: topo-esquerdo, topo-direito, baixo-direito, baixo-esquerdo")

while True:
    cv2.imshow("Selecione 4 pontos (TL, TR, BR, BL)", imagem_copia)
    if len(pontos) == 4:
        break
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

# Transformação
pts = np.array(pontos, dtype="float32")
def ordenar_pontos(p):
    ret = np.zeros((4, 2), dtype="float32")
    soma = p.sum(axis=1)
    diff = np.diff(p, axis=1)

    ret[0] = p[np.argmin(soma)]
    ret[2] = p[np.argmax(soma)]
    ret[1] = p[np.argmin(diff)]
    ret[3] = p[np.argmax(diff)]
    return ret

ordenado = ordenar_pontos(pts)
(tl, tr, br, bl) = ordenado

larguraA = np.linalg.norm(br - bl)
larguraB = np.linalg.norm(tr - tl)
alturaA = np.linalg.norm(tr - br)
alturaB = np.linalg.norm(tl - bl)

largura = int(max(larguraA, larguraB))
altura = int(max(alturaA, alturaB))

destino = np.array([
    [0, 0],
    [largura - 1, 0],
    [largura - 1, altura - 1],
    [0, altura - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(ordenado, destino)
corrigida = cv2.warpPerspective(imagem, M, (largura, altura))

cv2.imshow("Imagem Corrigida", corrigida)
cv2.imwrite("Imagem_corrigida.jpg", corrigida)
cv2.waitKey(0)
cv2.destroyAllWindows()
