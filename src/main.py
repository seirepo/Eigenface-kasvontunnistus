from tkinter import Tk
from app import App
from ui.ui import UI

def main():
    app = App()
    #app.suorita()
    window = Tk()
    window.title("test")

    ui = UI(window, app)
    ui.start()

    window.mainloop()

if __name__ == "__main__":
    main()
