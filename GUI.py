import os
import tkinter as tk
from tkinter import PhotoImage, filedialog,Text,Label, Canvas
import tkinter.font as tkFont
from PIL import ImageTk,Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Load trained model
new_model = tf.keras.models.load_model('saved_model/my_model')
batch_size = 32
img_height = 300
img_width = 300
class_names = ['brown_spot', 'normal', 'tungro']


window = tk.Tk()
window.title('PADDY DISEASE DETECTION')
window.geometry("1280x720")
filename ="null"

apps = [] 

# Open image function, with resizer to fit into the application
def add_App():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",title="Select File",
    filetypes= (("all files","*.*"),("exe","*.exe")))
    
    apps.append(filename)
    
    for app in apps:
        img = Image.open(app)
        img_resized = img.resize((300, 225), Image.BILINEAR)
        photo = (ImageTk.PhotoImage(img_resized))
        label_picture = tk.Label(window, image=photo, height=300, width=225)
        label_picture.photo = photo 
        #label_picture['image'] = label_picture.photo
        label_picture.place(relx=0.5, rely= 0.5, anchor = 'center')
        #label_picture.pack() 

    return filename 
 

# Use trained model to recognize input image function
def run_App():
    global filename
    global labeloutput
    img = keras.preprocessing.image.load_img(
    filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    Output = (
    "Type of Disease {}" # with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    labeloutput = Label(frame,text=Output,bg="white")
    labeloutput.place(relx=0.5, rely= 0.8, anchor = 'center')

    #labeloutput.pack()
 
 # Clear previous output function   
def remove_text():
	labeloutput.config(text="")


frame = tk.Frame(window,bg="white")
frame.place(relwidth=0.8,relheight=0.8,relx=0.1,rely=0.1)

# Output display
labeloutput = Label(frame,text="",bg="white")

fontStyle = tkFont.Font(family="Arial",size=20)
labeltitle = Label(frame,text="PADDY DETECTION" ,font= fontStyle,bg="white")
labeltitle.pack()

line = tk.Frame(frame, height=1, width=550, bg="grey80", relief='groove')
line.pack()

# Open Image button
openFile = tk.Button(frame,text="Select Image",padx=10,pady=5,fg="white",bg="#263D42",  command=add_App  )
openFile.pack(pady=10)
RunApp = tk.Button(frame,text="Run Application",padx=10,pady=5,fg="white",bg="#263D42",  command=lambda: [remove_text(), run_App()]  )
RunApp.pack()

window.mainloop()
    