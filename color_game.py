# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:50:10 2019

@author: abhiram_ch_v_n_s
"""

from tkinter import *

root = Tk()

root.title("color game")

button_frame = Frame(root)
button_frame.config(borderwidth=10, padx=150, pady=150)
#button_frame.grid(padx=150, pady = 150)



Button(root, text="blue", fg='blue', height = 2, width=10).grid(row=0,column=0)
Button(root, text="black", fg='black',height = 2, width=10).grid(row=0,column=1)
Button(root, text="green", fg='green',height = 2, width=10).grid(row=1,column=0)
Button(root, text="red", fg='red',height = 2, width=10).grid(row=1,column=1)





root.mainloop()