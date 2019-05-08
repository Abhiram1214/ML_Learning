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
    
        
        
        
        
contact = address_book()        

contact.add_contact("kaki", 87779)
contact.display_contacts()
contact.modify_contact('abhi')

