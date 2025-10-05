import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Imagens")

        # Variáveis
        self.image_paths = []
        self.current_index = 0
        self.current_image = None
        self.folder_path = ""

        # Campo de entrada do caminho
        self.path_entry = tk.Entry(root, width=50)
        self.path_entry.pack(pady=10)

        # Botão para escolher pasta
        browse_button = tk.Button(root, text="Escolher Pasta", command=self.load_folder)
        browse_button.pack(pady=5)

        # Canvas para imagem
        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

        # Botões de classificação
        button_frame = tk.Frame(root)
        button_frame.pack()

        buraco_btn = tk.Button(button_frame, text="Buraco", width=12, command=lambda: self.move_image("buracos"))
        buraco_btn.grid(row=0, column=0, padx=5)

        rachao_btn = tk.Button(button_frame, text="Rachão", width=12, command=lambda: self.move_image("rachaos"))
        rachao_btn.grid(row=0, column=1, padx=5)

        boa_btn = tk.Button(button_frame, text="Boa", width=12, command=lambda: self.move_image("boas"))
        boa_btn.grid(row=0, column=2, padx=5)

        descartar_btn = tk.Button(button_frame, text="Descartar", width=12, command=lambda: self.move_image("descartar"))
        descartar_btn.grid(row=0, column=3, padx=5)

    def load_folder(self):
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            return

        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, self.folder_path)

        # Lista de imagens
        self.image_paths = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0

        if self.image_paths:
            self.show_image()
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada na pasta.")

    def show_image(self):
        if self.current_index < len(self.image_paths):
            img_path = self.image_paths[self.current_index]
            img = Image.open(img_path)
            img.thumbnail((500, 500))  # Reduz tamanho para caber na tela
            self.current_image = ImageTk.PhotoImage(img)
            self.canvas.config(image=self.current_image)
        else:
            messagebox.showinfo("Fim", "Todas as imagens foram classificadas.")
            self.canvas.config(image="")

    def move_image(self, category):
        if self.current_index < len(self.image_paths):
            img_path = self.image_paths[self.current_index]

            # Cria a pasta de destino, se não existir
            dest_folder = os.path.join(self.folder_path, category)
            os.makedirs(dest_folder, exist_ok=True)

            # Move a imagem
            shutil.move(img_path, os.path.join(dest_folder, os.path.basename(img_path)))

            self.current_index += 1
            self.show_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
