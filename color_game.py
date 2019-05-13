# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:50:10 2019

@author: abhiram_ch_v_n_s
"""

from tkinter import *
from tkinter import Frame

root = Tk()

root.title("color game")

button_frame = Frame(root)
button_frame.pack(side=RIGHT)

Button(button_frame, text="blue", fg='blue', height = 2, width=10).grid(row=0,column=0)
Button(button_frame, text="black", fg='black',height = 2, width=10).grid(row=0,column=1)
Button(button_frame, text="green", fg='green',height = 2, width=10).grid(row=1,column=0)
Button(button_frame, text="red", fg='red',height = 2, width=10).grid(row=1,column=1)

button_frame.config(bg='cyan')







root.mainloop()