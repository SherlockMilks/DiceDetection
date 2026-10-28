import tkinter as tk
from gui import DiceGUI


def main():
    root = tk.Tk()
    app = DiceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()