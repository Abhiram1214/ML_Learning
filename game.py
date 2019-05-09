# -*- coding: utf-8 -*-
"""
Created on Thu May  9 04:31:02 2019

@author: abhiram_ch_v_n_s

Intro: practising python classes by creating a address book
"""

class address_book:
    
    def __init__(self):
        self.contacts_list = {}
    
    def add_contact(self, name, number):
        self.name = name
        self.number = number
        
        self.contacts_list[self.name] = self.number
        
        self.display_contacts()
    
    def display_contacts(self):
        
        for key, value in self.contacts_list.items():
            print(key)
            print(value)            
            print("------------")
            
            self.options()
        
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
























