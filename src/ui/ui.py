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
        #self.root.grid_columnconfigure(1, weight=1)
        #self.root.grid_columnconfigure(0, weight=1)
        self.people_frame = ttk.Frame(master=self.root)#, width=280)
        self.people_frame.grid(row=0, column=0, padx=50, sticky="nw")

        label_people = ttk.Label(master=self.people_frame, text="Henkilöt")
        label_people.grid(row=0, column=0, columnspan=4)

        #self.app.load_data()
        #self.app.create_individuals()
        self.app.suorita()
        ppl = self.app.get_image_of_everyone()

        self.show_faces(ppl)

        #button_calc = ttk.Button(
        #    master=self.people_frame,
        #    text="laske eigenfacet",
        #    command=self.handle_button_click
        #)
        #button_calc.grid(row=11, column=0, columnspan=4, pady=10)

        self.middle_canvas = tkinter.Canvas(master=self.root, width=280, height=280)
        self.middle_canvas.grid(row=0, column=1, sticky="nw")

        self.test_im_frame = ttk.Frame(master=self.middle_canvas)
        #self.test_im_frame = ttk.Frame(master=self.root, width=280)
        self.test_im_frame.grid(row=0, column=0, sticky="nw")

        test_label = ttk.Label(master=self.middle_canvas, text="Testikuvat")
        test_label.grid(row=0, column=0, columnspan=5)

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
        tmp_canvas_col = 1
        tmp_canvas_row = 1
        for ind in ppl:
            tmp_canvas = tkinter.Canvas(master=self.middle_canvas)
            tmp_canvas.grid(row=tmp_canvas_row, column=tmp_canvas_col, padx=15)
            nearest_nbrs = ind.get_nearest_neighbor()
            # pitää käsitellä molemmat nearest_nbrs listan alkiot!
            test_item = nearest_nbrs[0]
            im = test_item[0].reshape((64,64))
            nearest_id = test_item[1]
            im = np.uint8(im*255)
            im = ImageTk.PhotoImage(image=Image.fromarray(im))
            txt = "image of\n" + str(ind.get_id())
            label = ttk.Label(
                master=tmp_canvas,
                image=im,
                #text=str(ind.get_id()),
                text=txt,
                compound="top"
            )
            label.image = im
            label.grid(row=0, column=0, padx=5)

            nearest_im = self.app.get_image_by_id(nearest_id)
            nearest_im = nearest_im.reshape((64,64))
            nearest_im = np.uint8(nearest_im*255)
            nearest_im = ImageTk.PhotoImage(image=Image.fromarray(nearest_im))
            txt1 = "identified as\n" + str(nearest_id)
            label_nearest = ttk.Label(
                master=tmp_canvas,
                image=nearest_im,
                text=txt1,
                compound="top"
            )
            label_nearest.image = nearest_im
            label_nearest.grid(row=0, column=1)

            tmp_canvas_row += 1
            if tmp_canvas_row % 9 == 0:
                tmp_canvas_col += 1
                tmp_canvas_row = 1
            #if r % 9 == 0:
            #    r = 1
            #    c +=4

        # toimii mutta liian sekava
        #for ind in ppl[:25]:
        #    nearest_nbrs = ind.get_nearest_neighbor()
        #    test_item = nearest_nbrs[0]
        #    im = test_item[0].reshape((64,64))
        #    nearest_id = test_item[1]
        #    im = np.uint8(im*255)
        #    im = ImageTk.PhotoImage(image=Image.fromarray(im))
        #    label = ttk.Label(
        #        master=self.middle_canvas,
        #        image=im,
        #        text=str(ind.get_id()),
        #        compound="top"
        #    )
        #    label.image = im
        #    label.grid(row=r, column=c)
        #    print("kuva1: ", r, c)
#
        #    nearest_im = self.app.get_image_by_id(nearest_id)
        #    nearest_im = nearest_im.reshape((64,64))
        #    nearest_im = np.uint8(nearest_im*255)
        #    nearest_im = ImageTk.PhotoImage(image=Image.fromarray(nearest_im))
        #    label_nearest = ttk.Label(
        #        master=self.middle_canvas,
        #        image=nearest_im,
        #        text=str(nearest_id),
        #        compound="top"
        #    )
        #    label_nearest.image = nearest_im
        #    label_nearest.grid(row=r, column=c+1)
        #    print("kuva2: ", r, c+1)
#
        #    r += 1
        #    if r % 9 == 0:
        #        r = 1
        #        c +=4

    def image_to_label(self, master, image, text=""):
        im = image.reshape((64,64))
        im = np.uint8(im*255)
        im = ImageTk.PhotoImage(image=Image.fromarray(im))
        label = ttk.Label(
            master=master,
            image=im,
            text=text,
            compound="top"
        )
        label.image = image
        return label

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()

    def show_images(self):
        pass
