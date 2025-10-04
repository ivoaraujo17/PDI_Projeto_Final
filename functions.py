import os
import cv2
import numpy as np


# Função: aplicar pré-processamento
def preprocess(img):
    # Redimensionar para padronizar
    img = cv2.resize(img, (224, 224))

    # 1. Remoção de ruído sal e pimenta com filtro de mediana
    denoised = cv2.medianBlur(img, 3)

    # 2. Conversão p/ escala de cinza para aplicar Sobel
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # 3. Sobel X e Sobel Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 4. Magnitude da borda (combinação)
    sobel_comb = np.sqrt(sobelx**2 + sobely**2)
    sobel_comb = np.uint8(255 * sobel_comb / np.max(sobel_comb))

    return denoised, sobel_comb


def load_and_resize_images(folder_path='dataset', size=(512, 512)):
    # Pastas/classes
    classes = ["buracos", "rachaduras", "boas"]

    # Listas para armazenar dados
    imagens = []
    imgs_mask = []
    labels = []

    for label, classe in enumerate(classes):
        pasta = os.path.join(folder_path, classe)
        for file in os.listdir(pasta):
            if file.endswith((".jpg", ".png", ".jpeg")):
                caminho = os.path.join(pasta, file)
                img = cv2.imread(caminho)
                img = cv2.resize(img, size) 

                denoised, mask = preprocess(img)

                imagens.append(denoised)
                imgs_mask.append(mask)
                labels.append(label)

    print(f"Total de imagens: {len(imagens)}")
    print(f"Total de labels: {len(labels)}")
    
    return np.array(imagens), np.array(imgs_mask), np.array(labels)