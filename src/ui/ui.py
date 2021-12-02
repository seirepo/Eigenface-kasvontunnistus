from tkinter import ttk
import tkinter
from app import App
from PIL import Image, ImageTk

class UI:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.left_pane = None
        self.people = []

    def start(self):


        self.left_pane = ttk.Frame(master=self.root, width=280)
        self.left_pane.pack(anchor="sw")
        label_people = ttk.Label(master=self.left_pane, text="People")
        label_people.grid(row=0, column=0, columnspan=4)

        self.app.create_individuals()
        ppl = self.app.get_images_of_people()
        x = 1
        y = 0
        im = 1
        for person in ppl:
            image = ImageTk.PhotoImage(image=Image.fromarray(person))
            self.people.append(image)
            label = ttk.Label(self.left_pane, image=image, text=str(im), compound="top")
            label.image = image
            label.grid(row=x, column=y)
            x += 1
            if x % 11 == 0:
                x = 1
                y += 1
            im += 1

        button_calc = ttk.Button(
            master=self.left_pane,
            text="laske eigenfacet",
            command=self.handle_button_click
        )
        button_calc.grid(row=11, column=0, columnspan=4)

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()

    def show_images(self):
        pass
