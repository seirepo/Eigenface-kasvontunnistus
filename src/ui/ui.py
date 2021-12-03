from tkinter import ttk
import tkinter
from app import App
from PIL import Image, ImageTk
import numpy as np

class UI:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.main_pane = None
        self.people = []

    def start(self):


        self.main_pane = ttk.Frame(master=self.root, width=280)
        #self.main_pane.pack(anchor="sw")
        self.main_pane.grid(row=0, column=0)
        label_people = ttk.Label(master=self.main_pane, text="People")
        label_people.grid(row=0, column=0, columnspan=4)

        self.app.create_individuals()
        ppl = self.app.get_image_of_everyone()
        x = 1
        y = 0
        for pair in ppl:
            id = pair[0]
            image = np.uint8(pair[1]*255)
            image = ImageTk.PhotoImage(image=Image.fromarray(image))
            self.people.append(image)
            label = ttk.Label(self.main_pane, image=image, text=str(id), compound="top")
            label.image = image
            label.grid(row=x, column=y)
            x += 1
            if x % 11 == 0:
                x = 1
                y += 1

        button_calc = ttk.Button(
            master=self.main_pane,
            text="laske eigenfacet",
            command=self.handle_button_click
        )
        button_calc.grid(row=11, column=0, columnspan=4)

        self.middle_pane = ttk.Frame(master=self.root, width=280)
        self.middle_pane.grid(row=0, column=1)
        test_label = ttk.Label(master=self.middle_pane, text="TESTI")
        test_label.grid(row=0, column=0)

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()

    def show_images(self):
        pass
