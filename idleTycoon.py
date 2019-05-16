#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:03:45 2019

@author: abhiram

description: Python Idle Tycoon
"""

printDivider = '============================================='

class Store():
    Money = 25.00
    Day = 1
    StoreList = []
    
    def __init__(self, storename, storeprofit, storecost):
        
        self.storename = storename
        self.storeprofit = storeprofit
        self.storecost = storecost
        self.storeCount = 0
        
        inital_data = ['Lemon Stand', 3, 1, self.storeCount]
        Store.StoreList.append(inital_data)
        
    def displayStore(self):

        for i in Store.StoreList:
            print(i)
            #print(self.profit)
    
    def display_purchase(self):
        for store_name in Store.StoreList:
            print("{} costs $ {}. Its profit is {}".format(store_name[0], store_name[1], store_name[2]))
          
    def DisplayGameInfo(self): 
        
        print("---------------------------------------------") 
        
        print("Money $" + str(Store.Money)) 
        print("---------------------------------------------") 
        print("Stores".ljust(25) + "Store Cost".ljust(15) + "Store Count") 
            
    
    def buyStore(self):
        print("your money" + str(Store.Money))
        self.display_purchase()
        purchase_store = int(input("which store do you want to buy "))
        
        if purchase_store == 1:
            if Store.StoreList[0][2] <= Store.Money:
                Store.Money = Store.Money - Store.StoreList[0][2]
                Store.StoreList[0][3] += 1
            else:
                print("insufficient funds")
        else:
            print("wrong input")
            
        
        self.DisplayGameInfo()
        
        
    def nextDay(self):
        Store.Money += self.storeprofit 
        
    def addStore(self):
        add_name = input("Enter the store name ")
        add_cost = input("Enter the store cost ")
        add_profit = input("Enter the store profit ")
        
        self.name = add_name
        self.profit = add_profit
        self.cost = add_cost
        self.storeCount += 1
        
        new_record = [self.name, self.profit, self.cost, self.storeCount]
        Store.StoreList.append(new_record)
        
store = Store("Lemonade Stand", 1.50, 3)     
#Store.StoreList.append(Store("Lemonade Stand", 1.50, 3)) 
store.addStore()
store.displayStore()
store.buyStore()


while True:
    store.displayStore()
    
    select_option = input("Available chaices are: (N)NextDay, (B)Buy Store, (Q)QUit")
    
    if select_option.lower() == 'n':
        store.nextDay()
    elif select_option.lower() == 'b':
        store.buyStore()
    elif select_option.lower() == 'a':
        store.addStore()
    else:
        break
        
    
    
    
    
    
    
    
    
    
