from Project import *

from tkinter import *
import tkinter.messagebox

root = Tk()
root.title("Mail Detection")
root.geometry("655x455")


def fun_predict():
    text_mail = [Entry_value.get()]
    predict_value = prediction_model(text_mail)
    tkinter.messagebox.showinfo("Prediction", predict_value)


root.minsize(600, 400)

f1 = Frame(root, bg="skyblue", borderwidth=9, relief=SUNKEN)
f1.pack(fill=Y)  # Box padding and filling
Label(f1, text="Spam-Ham Mail Detection Model", font="lucid 20 bold", pady=20, padx=50).pack()

# noinspection SpellCheckingInspection
f2 = Frame(root)
f2.pack(fill=Y)

Label(f2, text="Enter the mail here", font="comicsansms 15 bold").pack()

mail_value = StringVar()

Entry_value = Entry(f2, textvariable=mail_value, width=49, font="Arial 14")
Entry_value.insert(END, "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, "
                        "k? I've cried "
                        "enough today.")
Entry_value.pack()
Button(f2, text="Submit", height=2, width=15, command=fun_predict).pack()

root.mainloop()
