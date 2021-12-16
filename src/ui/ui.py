from tkinter import Frame, Radiobutton, StringVar
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
        self.people_frame = tkinter.Frame(master=self.root, width=280)#, height=280)
        self.people_frame.grid(row=0, column=0, padx=30, sticky="nw")

        ppl = self.app.get_image_of_everyone()
        self.show_faces(ppl)

        self.button_frame = tkinter.Frame(master=self.people_frame)#, width=150, height=150)
        self.button_frame.grid(column=1, columnspan=5, sticky="nw")

        label_people = tkinter.Label(master=self.people_frame, text="Henkilöt")
        label_people.grid(row=0, column=0, columnspan=4)

        self.app.classify()

        label_nbrs = tkinter.Label(master=self.button_frame, text="neighbors:")
        label_pnorm = tkinter.Label(master=self.button_frame, text="order of the norm:")
        label_nbrs.grid()
        vk = StringVar(self.button_frame, 1)
        vnorm = StringVar(self.button_frame, 1)
        values_k = {
            "1" : 1,
            "2" : 2,
            "3" : 3,
            "4" : 4,
            "5" : 5,
        }

        for (text, value) in values_k.items():
            Radiobutton(
                self.button_frame, text = text,
                variable = vk, value = value
            ).grid(row=0, column=value)

        label_pnorm.grid()
        for (text, value) in values_k.items():
            Radiobutton(
                self.button_frame, text = text,
                variable = vnorm, value = value
            ).grid(row=1, column=value)

        button_classify = tkinter.Button(
            master=self.button_frame, #self.people_frame,
            text="Classify",
            command=lambda: self.handle_button_click(int(vk.get()), int(vnorm.get()))
        )
        button_classify.grid(pady=10)

        self.middle_canvas = tkinter.Canvas(master=self.root, width=280, height=280)
        self.middle_canvas.grid(row=0, column=1, sticky="nw")

        self.test_im_frame = tkinter.Frame(master=self.middle_canvas)
        self.test_im_frame.grid(row=0, column=0, sticky="nw")

        test_label = tkinter.Label(master=self.middle_canvas, text="Testikuvat")
        test_label.grid(row=0, column=0, columnspan=16)

        ppl = self.app.get_individuals()
        self.show_test_images(ppl)

    def show_faces(self, ppl):
        y = 1
        x = 1
        for pair in ppl:
            id = pair[0]
            image = np.uint8(pair[1]*255).reshape((64,64))
            image = ImageTk.PhotoImage(image=Image.fromarray(image))
            self.people.append(image)
            label = tkinter.Label(
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
        tmp_canvas_col = 1
        tmp_canvas_row = 1
        for ind in ppl:
            tmp_canvas = tkinter.Canvas(master=self.middle_canvas)
            tmp_canvas.grid(row=tmp_canvas_row, column=tmp_canvas_col, padx=10, pady=5)
            nearest_nbrs = ind.get_nearest_neighbor()
            row = 0
            col = 0
            for item in nearest_nbrs:
                test_im = item["test_im"]
                test_im = test_im.reshape((64,64))
                test_im = np.uint8(test_im*255)
                test_im = ImageTk.PhotoImage(image=Image.fromarray(test_im))
                txt = "input: " + str(ind.get_id())
                label = tkinter.Label(
                    master=tmp_canvas,
                    image=test_im,
                    text=txt,
                    compound="top"
                )
                label.image = test_im
                label.grid(row=row, column=col, padx=5)

                nearest_id = item["nearest_id"]
                nearest_crds = item["nearest_im_crds"]
                nearest_im = self.app.get_projected_image(nearest_crds)
                nearest_im = nearest_im.reshape((64,64))
                nearest_im = np.uint8(nearest_im*255)
                nearest_im = ImageTk.PhotoImage(image=Image.fromarray(nearest_im))
                txt1 = "classified: " + str(nearest_id)
                label_nearest = tkinter.Label(
                    master=tmp_canvas,
                    image=nearest_im,
                    text=txt1,
                    compound="top"
                )
                label_nearest.image = nearest_im
                if nearest_id != ind.get_id():
                    label_nearest.configure(background='red')
                label_nearest.grid(row=row, column=col+1)
                row += 1
                col = 0
                tmp_canvas_col += 1
                if tmp_canvas_col % 17 == 0:
                    tmp_canvas_row += 1
                    tmp_canvas_col = 1

    def handle_button_click(self, k, p):
        print("luokitellaan...")
        print(f"valitut parametrit\nk ({type(k)}): {k}, p ({type(p)}): {p}")
        self.app.classify(k, p)
        self.show_test_images(self.app.get_individuals())
        print("done")
