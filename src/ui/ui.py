from tkinter import Checkbutton, Tk, ttk

class UI:
    def __init__(self, root):
        self.root = root
        self.entry = None

    def start(self):
        label = ttk.Label(master=self.root, text="Hello world!")
        
        button = ttk.Button(
            master=self.root,
            text="paina tästä",
            command=self.handle_button_click)
        
        self.entry = ttk.Entry(master=self.root)
        checkb = ttk.Checkbutton(master=self.root, text="checkkaa tää")
        radiob = ttk.Radiobutton(master=self.root, text="tällanen")
        
        label.grid(row=0, column=0, columnspan=2)
        button.grid(row=1, column=2)
        self.entry.grid(row=1, column=0, columnspan=2)
        checkb.grid(row=2, column=0)
        radiob.grid(row=2, column=1)

    def handle_button_click(self):
        value = self.entry.get()
        print(f"syötetty teksti: {value}")
