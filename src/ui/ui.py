from tkinter import ttk
import tkinter
from app import App
from PIL import Image, ImageTk

class UI:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.canvas = None
        self.im = []

    def start(self):
        label = ttk.Label(master=self.root, text="Testikäyttöliittymä")
        
        button_calc = ttk.Button(
            master=self.root,
            text="laske eigenfacet",
            command=self.handle_button_click
        )

        button_show = ttk.Button(
            master=self.root,
            text="näytä kuva",
            command=self.show_images
        )
        
        self.canvas = tkinter.Canvas(master=self.root, width=300, height=300)

        im = self.app.get_random_image()
        img = ImageTk.PhotoImage(image=Image.fromarray(im))
        panel = ttk.Label(self.root, image = img)
        panel.image = img

        label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        button_calc.grid(row=1, column=0, padx=5, pady=5)
        button_show.grid(row=1, column=1, padx=5, pady=5)
        self.canvas.grid(row=2, column=4, padx=5, pady=5)
        panel.grid(row=2, column=3, padx=5, pady=5)

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()

    def show_images(self):
        print("näytetään kuva")
        im = self.app.get_random_image()
        image = ImageTk.PhotoImage(image=Image.fromarray(im))
        self.im.append(image)
        self.canvas.create_image(20, 20, anchor="nw", image=image)
