from tkinter import Frame, Scrollbar, ttk
import tkinter
from app import App
from PIL import Image, ImageTk
import numpy as np

class UI:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.people_frame = None
        self.people = []

    def start(self):

        self.people_frame = ttk.Frame(master=self.root, width=280)
        self.people_frame.grid(row=0, column=0, padx=50, sticky="nw")
        label_people = ttk.Label(master=self.people_frame, text="Henkilöt")
        label_people.grid(row=0, column=0, columnspan=4)

        self.app.load_data()
        self.app.create_individuals()
        ppl = self.app.get_image_of_everyone()

        self.show_faces(ppl)

        button_calc = ttk.Button(
            master=self.people_frame,
            text="laske eigenfacet",
            command=self.handle_button_click
        )
        button_calc.grid(row=11, column=0, columnspan=4, pady=10)

        self.middle_canvas = tkinter.Canvas(master=self.root, width=280, height=280)
        self.middle_canvas.grid(row=0, column=1, sticky="nw")

        self.test_im_frame = ttk.Frame(master=self.middle_canvas, width=280)
        self.test_im_frame.grid(row=1, column=1, sticky="nw")

        test_label = ttk.Label(master=self.middle_canvas, text="Testikuvat")
        test_label.grid(row=0, column=0, columnspan=5)

        ppl = self.app.get_individuals()

        self.show_test_images(ppl)

        #vsb = Scrollbar(master=self.middle_canvas, orient="vertical", command=self.middle_canvas.yview)
        #vsb.grid(row=1, column=2, sticky="ns")
        #self.middle_canvas.configure(yscrollcommand=vsb.set, yscrollincrement=5)
        #self.middle_canvas.configure(scrollregion=self.middle_canvas.bbox("all"))

    def show_faces(self, ppl):
        y = 1
        x = 1
        for pair in ppl:
            id = pair[0]
            image = np.uint8(pair[1]*255).reshape((64,64))
            image = ImageTk.PhotoImage(image=Image.fromarray(image))
            self.people.append(image)
            label = ttk.Label(
                self.people_frame,
                image=image, text=str(id),
                compound="top"
            )
            label.image = image
            label.grid(row=x, column=y)
            y += 1
            if y % 6 == 0:
                y = 1
                x += 1

    def show_test_images(self, ppl):
        x = 0
        y = 1
        for individual in ppl:
            test_images = individual.get_test_images()
            id = individual.get_id()
            for test_image in test_images.T:
                test = test_image.reshape((64,64))
                image = np.uint8(test*255)
                image = ImageTk.PhotoImage(image=Image.fromarray(image))
                radiob = ttk.Label(
                    master=self.test_im_frame,
                    image=image, text=str(id), compound="top"
                )
                radiob.image = image
                radiob.grid(row=x, column=y)
                y += 1
                if y % 10 == 0:
                    y = 1
                    x += 1

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()

    def show_images(self):
        pass
