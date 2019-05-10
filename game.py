# -*- coding: utf-8 -*-
"""
Created on Thu May  9 04:31:02 2019

@author: abhiram_ch_v_n_s

Intro: practising python classes by creating a address book
"""

from tkinter import *

class address_book:
    def __init__(self, master):
        
        self.phone_book = {}
        self.master = master
        self.frame_header = Frame(master)
        
        Label(master, text="name").grid(row=0,column=0)
        Label(master, text="number").grid(row=1,column=0)
        Label(master, text="to be deleted").grid(row=4,column=0)
        Button(master, text="Add", command = self.my_contact).grid(row=3)
        Button(master, text="display", command = self.display_contacts).grid(row=3, column=2)
        Button(master, text="delete", command = self.delete_code).grid(row=3, column=3)
        
        
        self.e1 = Entry(master)
        self.e2 = Entry(master)
        self.d1 = Entry(master)
        
        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.d1.grid(row=4, column=1)


    def my_contact(self):
        self.name = self.e1.get()
        self.number = self.e2.get()
        
        name = self.name
        number = self.number
    
        self.phone_book[name] = number

    
    def display_contacts(self):
        
        #Label(master, text="Address book")
        #Label(master, text="-------------------")
        
        self.frame_header.grid(row = 5, column = 0)
        for key, value in self.phone_book.items():
            Label(self.frame_header, text=key).grid(rowspan=15)
            Label(self.frame_header, text=value).grid()
            


    def delete_code(self):
    
        if self.d1.get() in self.phone_book:
            del self.phone_book[self.d1.get()]
            #Label(master, text="deleted").grid(row=3,column=0)
        else:
            Label(master, text="not in phonebook").grid(row=3,column=0)

master = Tk()
contact = address_book(master)
master.mainloop()



































'''
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




from tkinter import *
from os import system

phone_book = {}

def my_contact():
    
   name = e1.get()
   number = e2.get()
    
   phone_book[name] = number


def display_contacts():
    
    tkinter.Label(master, text="Address book")
    tkinter.Label(master, text="-------------------")
    for key, value in phone_book.items():
        tkinter.Label(master, text=key).grid(rowspan=15)
        tkinter.Label(master, text=value).grid()
        
           
def delete_entry():
    window = Tk()
    window.title("delete contact")
    d1 = Entry(window)    
    d1.grid(row=0)
    tkinter.Button(window, text='delete', command=delete_code).grid(row=1)  
    
    
    
def delete_code():
    
    if d1.get() in phone_book:
        del phone_book[d1.get()]
        tkinter.Label(master, text="deleted").grid(row=3,column=0)
    else:
        tkinter.Label(master, text="not in phonebook").grid(row=3,column=0)

    
    

master = Tk()
master.title("add contacts")


e1 = Entry(master)
e2 = Entry(master)
d1 = Entry(master)


tkinter.Label(master, text="name").grid(row=0,column=0)
e1.grid(row=0, column=1)

tkinter.Label(master, text="number").grid(row=1,column=0)
e2.grid(row=1, column=1)

tkinter.Label(master, text="to be deleted").grid(row=4,column=0)
d1.grid(row=4, column=1)

tkinter.Button(master, text="Add", command = my_contact).grid(row=3)

tkinter.Button(master, text="display", command = display_contacts).grid(row=3, column=2)

tkinter.Button(master, text="delete", command = delete_code).grid(row=3, column=3)

mainloop()

'''







