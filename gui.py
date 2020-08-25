import numpy as np
import cv2
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image
from PIL import ImageTk
from recognize import recognize
from extract_embeddings import extract_embeddings
from train_model import train_model
import os
from imutils import paths
from datetime import datetime
from ttkthemes import ThemedTk
import shutil

#initialize the model
Recognize = recognize(0.5,0.8)
#initialize the embedding model
extractEmbed = extract_embeddings()

#Set up GUI
#window = tk.Tk()  #Makes main window
window = ThemedTk(theme="equilux")
window.title("Facial Recognition Surveillance System")
window.iconbitmap("logo3_icon.ico")
window.config(background = "#292929")#"#121212")#"#242424")
#print(window.style)
#style.configure("BW.TLabel", foreground="black", background="white")
#window.config(background="#FFFFFF")
#window.bind('<<ThemeChanged>>', lambda event: print('theme changed in root and across all widgets!'))
'''
Style = ttk.Style()
print(Style.theme_use())

Style.theme_use("clam")
print(Style.theme_use())
'''
#print(ttk.Style().theme_names())
style = ttk.Style()
style.theme_use("equilux")
print(style.lookup("TButton", "foreground", default="white"))
#number of images to be saved
maxNumImg = 20

#blacklist list
blacklist = list()

consecutiveAlert = False
#Graphics window

imageFrame = ttk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

#infoFrame = ttk.Frame(window)#,width=600, height=500)
#infoFrame.grid(row=0, column=5)#, padx=10, pady=20)
#info = tk.Text(infoFrame)
#s = ttk.Scrollbar( infoFrame, orient=tk.VERTICAL, command=info.yview)
info = tk.Text(window)
#info.grid(row=0, column=5)
s = ttk.Scrollbar( window, orient=tk.VERTICAL, command=info.yview)
#s.configure(height = 26)
s.grid(row = 0,column = 6, sticky = (tk.N,tk.S), pady=15 ,padx=(0,10))
info.configure(yscrollcommand=s.set,height = 26)
info.grid(row=0, column=4)#, pady=40)
info.configure(font = ("Calibri",11))

def updateText(text):
    global info
    info.config(state=tk.NORMAL)
    info.insert(tk.END,text)
    info.config(state=tk.DISABLED)

updateText("[INFO] Loaded.\n")

#Capture video frames
lmain = ttk.Label(imageFrame)
lmain.pack()
#lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(1)
gframe = startX = startY = endX = endY = None
imgCount = 0
def show_frame():
    global gframe,startX, startY, endX, endY,consecutiveAlert
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gframe = frame.copy()
    frame, (startX, startY, endX, endY),name,proba = Recognize.detect(frame)

    if name in blacklist:
        color = (0, 0, 255)
        if not consecutiveAlert:
            updateText("[ALERT] Blacklisted person "+name +" detected! at "+str(datetime.now())+"\n")
            consecutiveAlert = True
    else:
        color = (255, 0, 0)
        consecutiveAlert = False
    # draw the bounding box of the face along with the associated
    # probability
    if name is not None:
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) 

def createAddWindow(event,newUser):
    def addNew(event):
        global imgCount
        imgCount = 0
        previousImgCount = 0
        #print(Recognize.namePresent(name.get()))
        #if exist(name.get()):
        #print("newUser",newUser)
        if len(name.get())==0:
            warningLblText.set("Please enter the name!")
            warningLbl.grid()
            return
        if Recognize.namePresent(name.get()):
            #show warning label
            if newUser:
                warningLblText.set("The user already exist.")
                warningLbl.grid()
                return
            else:
                #imgCount = int(list(paths.list_images(os.path.sep.join(["dataset",name.get()])))[-1].split(os.path.sep)[-1].split(".")[-2])+1

                for path in list(paths.list_images(os.path.sep.join(["dataset",name.get()]))):
                    tmpImgCount = int(path.split(os.path.sep)[-1].split(".")[-2])
                    if tmpImgCount>imgCount:
                        imgCount = tmpImgCount
                        previousImgCount = imgCount

                print(imgCount,list(paths.list_images(os.path.sep.join(["dataset",name.get()]))))
        else:
            if not newUser:
                warningLblText.set("The user doesn't exist.")
                warningLbl.grid()
                return

            warningLbl.grid_remove()
            #to-do store


            
        def saveFaceLoop(nameStr):
            global gframe, startX, startY, endX, endY, imgCount
            face = gframe[startY:endY, startX:endX]
            path = os.path.sep.join(["dataset",nameStr,str(imgCount)+".jpg"])
            #if new person, create new folder
            if not os.path.isdir(os.path.sep.join(["dataset",nameStr])):
                os.mkdir(os.path.sep.join(["dataset",nameStr]))
            
            cv2.imwrite(path,face)
            print(path,imgCount,imgCount<=20)
            if imgCount-previousImgCount<=maxNumImg:
                imgCount+=1
                lmain.after(300,saveFaceLoop,nameStr)
            else:
                #extracting 128-D embeddings
                updateText("[INFO] Extracting embeddings\n")
                extractEmbed.extract()
                #train
                updateText("[INFO] Training model\n")
                train_model.train()
                #Reload model
                updateText("[INFO] Reloading model\n")
                Recognize.updateModel()
                if newUser:
                    updateText("[INFO] New user added:"+nameStr+"\n")
                else:
                    updateText("[INFO] User "+nameStr+" updated\n")

#           updateText("[NEW USER] new user added:"+name.get())
        saveFaceLoop(name.get())

        #todo start extraction and training

        
        #closing the subwindow
#            updateText("[NEW USER] new user added:"+name.get())
        addWin.destroy()
            
        
    global gframe
    addWin = tk.Toplevel()
    if newUser:
        addWin.title("Add New User")
    else:
        addWin.title("Add to Existing User")
    addWin.config(background = "#292929")#"#464646")
    #print(gframe.shape,(startX, startY, endX, endY))
    frame = gframe[startY-10:endY+10, startX-10:endX+10]
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    imgLbl = ttk.Label(addWin,image=imgtk)
    imgLbl.image = imgtk
    imgLbl.grid(row=0, column=0, columnspan = 3,pady = (10,0))
    label = ttk.Label(addWin, text="Enter Name:")
    label.grid(row= 1, column=0, padx=5,pady=10)
    label.config(background = "#292929")
    name = ttk.Entry(addWin)
    name.grid(row = 1, column = 1)
    name.bind("<Return>", addNew)
    name.config(background = "#292929")
    enterBtn = ttk.Button(addWin, text="Enter")
    enterBtn.bind("<Button-1>", addNew)
    enterBtn.grid(row = 1,column = 2,padx=5)
    warningLblText = tk.StringVar(value = "This name already exists. Please use another name.")
    warningLbl = ttk.Label(addWin, textvariable = warningLblText, foreground = "#CF6679",background= "#292929")
    warningLbl.grid(row = 2, column = 0)
    warningLbl.grid_remove()
    addWin.mainloop()

def createBlkListWindow(event):
    #print(Style.theme_use(),"2")
    def add(event):
        global blacklist
        if len(name.get())==0:
            warningLblText.set("Please enter the name!")
            warningLbl.grid()
            return
        if Recognize.namePresent(name.get()):
            blacklist.append(name.get())
            
        else:
            warningLblText.set("The user doesn't exist.")
            warningLbl.grid()


    def remove(event):
        global blacklist
        if len(name.get())==0:
            warningLblText.set("Please enter the name!")
            warningLbl.grid()
            return
        if Recognize.namePresent(name.get()):
            blacklist.remove(name.get())
        else:
            warningLblText.set("The user doesn't exist.")
            warningLbl.grid()

    blkWin = tk.Toplevel()
    blkWin.title("Blacklist user")
    blkWin.config(background = "#292929")
    lbl = ttk.Label(blkWin,text= "Enter name:")
    lbl.grid(row = 0,column = 0, padx=10)
    lbl.config(background= "#292929")
    name = ttk.Entry(blkWin)
    name.grid(row = 0,column = 1,padx=5,pady=5)
    addToListBtn = ttk.Button(blkWin, text="add to blacklist")
    addToListBtn.grid(row = 1,column = 0,padx=5,pady=5)
    addToListBtn.bind("<Button-1>", add)
    removeFromListBtn = ttk.Button(blkWin, text="remove from blacklist")
    removeFromListBtn.grid(row = 1,column = 1,padx= (0,5))
    removeFromListBtn.bind("<Button-1>", remove)
    warningLblText = tk.StringVar(value = "The user doesn't exist.")
    warningLbl = ttk.Label(blkWin, textvariable = warningLblText, foreground = "#CF6679",background= "#292929")
    warningLbl.grid(row = 2, column = 0)
    warningLbl.grid_remove()

    blkWin.mainloop()

def createRemoveWindow(event):

    def remove(event):
        if len(name.get())==0:
            warningLblText.set("Please enter the name!")
            warningLbl.grid()
            return
        if Recognize.namePresent(name.get()):
            shutil.rmtree(os.path.sep.join(["dataset",name.get()]))
            updateText("[INFO] User "+name.get()+" removed\n")

            #retrain & load model

            #extracting 128-D embeddings
            updateText("[INFO] Extracting embeddings\n")
            extractEmbed.extract()
            #train
            updateText("[INFO] Training model\n")
            train_model.train()
            #Reload model
            updateText("[INFO] Reloading model\n")
            Recognize.updateModel()
            removeWin.destroy()

        else:
            warningLblText.set("The user doesn't exist!")
            warningLbl.grid()
            return
            


    removeWin = tk.Toplevel()
    removeWin.title("Remove user")
    removeWin.config(background = "#292929")
    lbl = ttk.Label(removeWin,text= "Enter name:")
    lbl.grid(row = 0,column = 0, padx=10)
    lbl.config(background= "#292929")
    name = ttk.Entry(removeWin)
    name.grid(row = 0,column = 1,padx=5,pady=5)
    addToListBtn = ttk.Button(removeWin, text="remove")
    addToListBtn.grid(row = 1,column = 1,padx=5,pady=5)
    addToListBtn.bind("<Button-1>", remove)
    warningLblText = tk.StringVar(value = "The user doesn't exist!")
    warningLbl = ttk.Label(removeWin, textvariable = warningLblText, foreground = "#CF6679",background= "#292929")
    warningLbl.grid(row = 2, column = 0)
    warningLbl.grid_remove()


    removeWin.mainloop()

blacklistBtn = ttk.Button(window, text="Blacklist")
blacklistBtn.grid(row=1, column=0,pady=(0,10))
blacklistBtn.bind("<Button-1>", createBlkListWindow)
removeBtn = ttk.Button(window, text = "Remove existing user")
removeBtn.grid(row=1, column=1,pady=(0,10))
removeBtn.bind("<Button-1>", createRemoveWindow)
addExistingBtn = ttk.Button(window, text="Add to Existing User")
addExistingBtn.grid(row=1, column=2,pady=(0,10))
addExistingBtn.bind("<Button-1>", lambda event: createAddWindow(event,False))
addNewBtn = ttk.Button(window, text="Add New User")
addNewBtn.grid(row=1, column=3,pady=(0,10))
addNewBtn.bind("<Button-1>",  lambda event: createAddWindow(event,True))


#Slider window (slider controls stage position)
#sliderFrame = tk.Frame(window, width=600, height=100)
#sliderFrame.grid(row = 600, column=0, padx=10, pady=2)

#print("hola senora")

show_frame()  #Display 2
window.mainloop()  #Starts GUI