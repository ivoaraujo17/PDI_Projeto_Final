import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import exposure, color, filters
import numpy as np


def load_and_resize_images(folder_path='data/images', size=(512, 512)):
    """
    Lê todas as imagens da pasta especificada e redimensiona para o tamanho fornecido.
    Retorna um dicionário de imagens PIL redimensionadas, com a chave sendo o nome do arquivo sem extensão.
    """
    images = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            if "vlcsnap" not in filename:
                continue
            key = os.path.splitext(filename)[0]
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
                images[key] = img_resized
    return images


def load_labels(folder_path='data/labels'):
    """
    Lê todos os arquivos de label da pasta especificada.
    Retorna um dicionário com a chave sendo o nome do arquivo sem extensão e o valor sendo o conteúdo do label.
    """
    labels = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.txt', '.csv', '.json')):
            if "vlcsnap" not in filename:
                continue
            key = os.path.splitext(filename)[0]
            label_path = os.path.join(folder_path, filename)
            with open(label_path, 'r', encoding='utf-8') as f:
                labels[key] = f.read()
    return labels


def plot_image_with_polygons(img, label_text):
    """
    Plota a imagem com os polígonos desenhados a partir do texto do label.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    width, height = img_copy.size

    # Cada linha corresponde a um objeto anotado
    for line in label_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        class_id = int(parts[0])  # primeira coluna é a classe
        coords = list(map(float, parts[1:]))

        # Transforma coordenadas normalizadas em pixels
        polygon = [(coords[i] * width, coords[i+1] * height) for i in range(0, len(coords), 2)]

        # Desenha o polígono
        draw.polygon(polygon, outline="red", width=2)

        # (opcional) escreve o ID da classe no primeiro ponto
        draw.text(polygon[0], str(class_id), fill="yellow")

    # Plota a imagem
    plt.figure(figsize=(6, 6))
    plt.imshow(img_copy)
    plt.axis("off")
    plt.show()


def sobel_edges(img_pil):
    """
    Aplica o filtro Sobel combinado (horizontal + vertical) e plota resultado.
    """
    # Converte para tons de cinza
    img_gray = np.array(img_pil.convert("L")) / 255.0  # normaliza 0-1

    # Aplica Sobel horizontal e vertical
    sobel_h = filters.sobel_h(img_gray)
    sobel_v = filters.sobel_v(img_gray)

    # Combina as bordas (magnitude)
    sobel_combined = np.hypot(sobel_h, sobel_v)  # sqrt(sobel_h^2 + sobel_v^2)
    sobel_combined = np.clip(sobel_combined, 0, 1)  # garante faixa 0-1

    # Plota comparação
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Original (Gray)")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(sobel_combined, cmap="gray")
    plt.title("Bordas (Sobel)")
    plt.axis("off")

    plt.show()

    # Converte para PIL, opcional
    sobel_pil = Image.fromarray((sobel_combined * 255).astype(np.uint8))
    return sobel_pil


def equalize_image_rgb(img_pil):
    # Converte PIL -> NumPy e normaliza para [0,1]
    img = np.array(img_pil) / 255.0  

    # Converte para espaço YUV
    img_yuv = color.rgb2yuv(img)

    # Equaliza só o canal Y (luminância)
    img_yuv[..., 0] = exposure.equalize_hist(img_yuv[..., 0])

    # Converte de volta para RGB
    img_eq = color.yuv2rgb(img_yuv)

    # Garante faixa [0,255] e tipo uint8
    img_eq = np.clip(img_eq * 255, 0, 255).astype(np.uint8)

    # Volta para PIL
    eq_pil = Image.fromarray(img_eq)

    # Exibe comparação
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_pil)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(eq_pil)
    plt.title("Equalizada (RGB)")
    plt.axis("off")
    plt.show()

    return eq_pil


