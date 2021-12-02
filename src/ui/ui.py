from tkinter import ttk
from app import App

class UI:
    def __init__(self, root, app):
        self.root = root
        self.app = app

    def start(self):
        label = ttk.Label(master=self.root, text="Testikäyttöliittymä")
        
        button = ttk.Button(
            master=self.root,
            text="laske eigenfacet",
            command=self.handle_button_click)
        
        label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        button.grid(row=1, column=0, padx=5, pady=5)

    def handle_button_click(self):
        print("lasketaan...")
        self.app.suorita()
