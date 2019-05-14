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
        
        
    def displayStore(self):
        for i in Store.StoreList:
            print(i)
    
    def buyStore(self):
        Store.Money = Store.Money - 3.00
        self.displayStore()
        
    def nextDay(self):
        Store.Money += self.storeprofit 
        
store = Store("Lemonade Stand", 1.50, 3)     
#Store.StoreList.append(Store("Lemonade Stand", 1.50, 3)) 
Store.StoreList.append(Store("Record Stand", 1.50, 3)) 
store.displayStore()

'''
while True:
    store.displayStore()
    
    select_option = input("Available chaices are: (N)NextDay, (B)Buy Store, (Q)QUit")
    
    if select_option.lower() == 'n':
        store.nextDay()
    elif select_option.lower() == 'b':
        store.buyStore()
    else:
        break
        '''
    
    
    
    
    
    
    
    
    
