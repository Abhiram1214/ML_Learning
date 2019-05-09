# -*- coding: utf-8 -*-
"""
Created on Thu May  9 04:31:02 2019

@author: abhiram_ch_v_n_s

Intro: practising python classes by creating a address book
"""

import tkinter

class address_book:
    
    def __init__(self):
        self.contacts_list = {}
    
    def add_contact(self, name, number):
        self.name = name
        self.number = number
        
        self.contacts_list[self.name] = self.number
        
        self.options()
    
    def display_contacts(self):
        
        for key, value in self.contacts_list.items():
            print(key)
            print(value)            
            print("------------")
            
    def modify_contact(self, update_name):
        
        if update_name in self.contacts_list:
            new_contact = input("please update the name:  ")
            update_num = input("please update the number:  ")
            
            del self.contacts_list[update_name]
            
            self.contacts_list[new_contact] = update_num
            
            print("updated contacted list"+'\n')  
            self.display_contacts()
        else:
            print("contact not in the book")            
    
    def options(self):
        print("Enter 1 for adding contacts \n")
        print("Enter 2 for displaying contacts \n")
        print("Enter 3 for modifying contacts \n")
        option = int(input("please enter an option"))
             
                
        if option == 1:
           option_name = input("Enter the name: ")
           option_num = input("enter a number: ")
           self.add_contact(option_name, option_num)
        elif option == 2:
            self.display_contacts()
        elif option== 3:
            option_mod_name = input("Enter the name: ")
            #option_mod_num = input("enter a number")
            self.modify_contact(option_mod_name)
        else:
            print("incorrect")
            
        
        
contact = address_book()        

contact.add_contact("abhi", 1255)
contact.display_contacts()
contact.modify_contact('abhi')


contact.options()
contact.add_contact("abhi", 1255)
contact.add_contact("kaki", 111)




'''


#--------Tkinter--------------

window = tkinter.Tk()
window.title("Contacts book")
label = tkinter.Label(window, text="Hello world").pack()


btn1 = tkinter.Button(top_frame, text = "Button1", fg = "red").pack()
btn2 = tkinter.Button(top_frame, text = "Button2", fg = "green").pack()


top_frame = tkinter.Frame(window).pack()
bottom_frame = tkinter.Frame(window).pack(side='bottom')

window.mainloop()









import tkinter

window = tkinter.Tk()
window.title("GUI")

tkinter.Label(window, text = "Enter contact name").grid(row=0)
tkinter.Entry(window).grid(row=0, column=1)

tkinter.Label(window, text = 'Enter contact number').grid(row=1)
tkinter.Entry(window).grid(row=1, column=1)
tkinter.Checkbutton(window, text = "Keep me logged in").grid(columnspan=2)

tkinter.Button(window, text = "submit").grid(row=3, column=1)
window.mainloop()

'''


from tkinter import *

phone_book = {}

def my_contact():
    
   name = e1.get()
   number = e2.get()
    
   phone_book[name] = number


def display_contacts():
    
    tkinter.Label(master, text="Address book")
    tkinter.Label(master, text="-------------------")
    for key, value in phone_book.items():
        tkinter.Label(master, text=key).grid()
        tkinter.Label(master, text=value).grid()
        
           


master = Tk()
master.title("add contacts")

e1 = Entry(master)
e2 = Entry(master)

tkinter.Label(master, text="name").grid(row=0,column=0)
e1.grid(row=0, column=1)

tkinter.Label(master, text="number").grid(row=1,column=0)
e2.grid(row=1, column=1)

tkinter.Button(master, text="Add", command = my_contact).grid(row=3)

tkinter.Button(master, text="display", command = display_contacts).grid(row=3, column=2)

mainloop()


