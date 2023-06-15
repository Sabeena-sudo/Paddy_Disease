import tkinter as tk
from tkinter.ttk import *
from tkinter import *
from PIL import Image, ImageTk
from customtkinter import *
import numpy as np
import tensorflow as tf
from tensorflow import keras


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        # Load trained model
        self.new_model = tf.keras.models.load_model('saved_model/my_model')
        self.batch_size = 32
        self.img_height = 300
        self.img_width = 300
        self.class_names = ['brown_spot', 'normal', 'tungro']

        self.imageplaceholder = r'uploadImage.png'
        self.title("Daddy Disease Detection")
        self.geometry("700x400")
        self.resizable(False, False)
        self.columnconfigure(0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0)
        self.rowconfigure(1)
        self.rowconfigure(2, weight=1)
        CTkLabel(self, text="DADDY DISEASE", font=CTkFont(size=35), text_color="black").grid(row=0, column=0,
                                                                                             columnspan=2)
        CTkLabel(self, text="Detection System", text_color="black").grid(row=1, column=0, columnspan=2)

        # Start submenu part
        submenu = Frame(self)
        submenu.grid(row=2, column=0, padx=10)
        submenu.rowconfigure(0, weight=1)
        submenu.rowconfigure(1, weight=1)
        submenu.rowconfigure(2, weight=1)
        submenu.rowconfigure(3, weight=1)
        # C:\apikk\project\Paddy_Disease\uploadImage.png
        image = Image.open(self.imageplaceholder)
        resizedimage = image.resize((80, 80), Image.LANCZOS)

        newimage = ImageTk.PhotoImage(resizedimage)
        self.imagepicker = Button(submenu, borderwidth=0, image=newimage, command=self.addpicture)
        self.imagepicker.grid(row=0, column=0)
        self.imagepicker.image = newimage

        CTkButton(submenu, text="Start", command=self.predict).grid(row=1, column=0, pady=20)

        CTkButton(submenu, text="Clear", command=self.clear).grid(row=2, column=0, pady=10)
        CTkButton(submenu, text="Exit", command=exit).grid(row=3, column=0)
        # End submenu part

        # Start main content part

        mainpart = Frame(self)
        mainpart.grid(row=2, column=1)

        mainpart.rowconfigure(0)
        mainpart.rowconfigure(1)
        mainpart.rowconfigure(2)
        mainpart.columnconfigure(0, weight=2)
        mainpart.columnconfigure(1, weight=1)
        mainpart.columnconfigure(2, weight=1)
        image = Image.open(self.imageplaceholder)
        resizedimage = image.resize((200, 200), Image.LANCZOS)
        newimage = ImageTk.PhotoImage(resizedimage)
        self.imgw = CTkLabel(mainpart, image=newimage, text="")
        self.imgw.grid(row=0, column=0, columnspan=3)

        CTkLabel(mainpart, text="Paddy status", text_color="black").grid(row=1, column=1, sticky="w")
        CTkLabel(mainpart, text="Type of disease", text_color="black").grid(row=2, column=1, sticky="w")
        CTkLabel(mainpart, text=" ", text_color="black").grid(row=2, column=0)
        self.btnpaddystatus = CTkLabel(mainpart, text="", text_color="black")
        self.btnpaddystatus.grid(row=1, column=3)

        self.btntypedisease = CTkLabel(mainpart, text="", text_color="black")
        self.btntypedisease.grid(row=2, column=3)

    def addpicture(self):
        self.filepath = filedialog.askopenfilename(title="Select an image", filetypes=[("Image ", ".png .jpeg .jpg")])
        if not self.filepath:
            return
        image = Image.open(self.filepath)
        resizedimage = image.resize((80, 80), Image.LANCZOS)
        newimage = ImageTk.PhotoImage(resizedimage)
        self.imagepicker.configure(image=newimage)
        self.imagepicker.image = newimage

        resizedimagel = image.resize((200, 200), Image.LANCZOS)
        newimagel = ImageTk.PhotoImage(resizedimagel)
        self.imgw.configure(image=newimagel)
        self.imgw.image = newimagel

    def predict(self):
        if not self.filepath:
            return
        img = keras.preprocessing.image.load_img(
            self.filepath, target_size=(self.img_height, self.img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = self.new_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        Output = (
            "Type of Disease {}"  # with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score)))
        self.btnpaddystatus.configure(text=("Normal" if self.class_names[np.argmax(score)] == "normal" else "Disease"))
        self.btntypedisease.configure(text=self.class_names[np.argmax(score)])

    def clear(self):
        self.btnpaddystatus.configure(text="")
        self.btntypedisease.configure(text="")
        self.filepath = ""

        image = Image.open(self.imageplaceholder)
        resizedimage = image.resize((80, 80), Image.LANCZOS)
        newimage = ImageTk.PhotoImage(resizedimage)
        self.imagepicker.configure(image=newimage)
        self.imagepicker.image = newimage

        resizedimagel = image.resize((200, 200), Image.LANCZOS)
        newimagel = ImageTk.PhotoImage(resizedimagel)
        self.imgw.configure(image=newimagel)
        self.imgw.image = newimagel


def main():
    window = MainWindow()
    window.mainloop()


if __name__ == "__main__":
    main()
