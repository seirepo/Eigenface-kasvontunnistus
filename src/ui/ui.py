from tkinter import ttk
import tkinter
from app import App
from PIL import Image, ImageTk
import numpy as np

class UI:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.left_pane = None
        self.people = []

    def start(self):

        self.left_pane = ttk.Frame(master=self.root, width=280)
        self.left_pane.grid(row=0, column=0, padx=50)
        label_people = ttk.Label(master=self.left_pane, text="Henkilöt")
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
            label = ttk.Label(self.left_pane, image=image, text=str(id), compound="top")
            label.image = image
            label.grid(row=x, column=y)
            x += 1
            if x % 11 == 0:
                x = 1
                y += 1

        button_calc = ttk.Button(
            master=self.left_pane,
            text="laske eigenfacet",
            command=self.handle_button_click
        )
        button_calc.grid(row=11, column=0, columnspan=4)

        self.middle_pane = ttk.Frame(master=self.root, width=280)
        self.middle_pane.grid(row=0, column=1)
        test_label = ttk.Label(master=self.middle_pane, text="Testikuvat")
        test_label.grid(row=0, column=0, columnspan=5)

        ppl = self.app.get_individuals()
        #ind = ppl[0]
        #ims = ind.get_test_images()
        #tims = ims[:,0].reshape(64,64)
        #im = np.uint8(tims*255)
        #image = ImageTk.PhotoImage(image=Image.fromarray(im))
        #radiob = ttk.Radiobutton(master=self.middle_pane, image=image, text=str(id), compound="top")
        #radiob.image = image
        #radiob.grid(row=1, column=0)
        x = 1
        y = 0
        for individual in ppl:
            test_images = individual.get_test_images()
            id = individual.get_id()
            for test_image in test_images.T:
                test = test_image.reshape((64,64))
                image = np.uint8(test*255)
                image = ImageTk.PhotoImage(image=Image.fromarray(image))
                radiob = ttk.Button(
                    master=self.middle_pane,
                    image=image, text=str(id), compound="top"
                )
                radiob.image = image
                radiob.grid(row=x, column=y)
                x += 1
                if x % 11 == 0:
                    x = 1
                    y += 1

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()

    def show_images(self):
        pass
