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
        
        self.canvas = tkinter.Canvas(master=self.root, width=300, height=300)

        label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        button_calc.grid(row=1, column=0, padx=5, pady=5)
        self.canvas.grid(row=2, column=4, padx=5, pady=5)

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()

    def show_images(self):
        pass
