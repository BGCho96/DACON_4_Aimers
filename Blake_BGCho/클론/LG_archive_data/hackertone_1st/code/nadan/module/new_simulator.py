import os
import pandas as pd
import numpy as np
import time
import math

class NewSimulator:
    def __init__(self):
        pass

    def get_score(self, df):
        stock = pd.read_csv('module/stock.csv')
        self.now_stock = np.array(stock.loc[0], dtype=np.float32)
        order = pd.read_csv('module/order.csv')
        order = np.array(order)
        self.order = order[:, 1:]
        self.MOL_queue_A = np.zeros([49, 4])
        self.MOL_queue_B = np.zeros([49, 4])

        self.blk_diffs = []

        self.df = np.array(df.loc[:, "Event_A":"MOL_B"])

        def calc(e, order):
            cutting(e)                
            gap = float(math.floor(self.now_stock[8+e] - order))
            self.blk_diffs.append(gap)
            self.now_stock[8+e] = gap
                
        def cutting(e):
            if e == 0:
                self.now_stock[8] += self.now_stock[4] * 506 * 0.851
                self.now_stock[4] = 0
            if e == 1:
                self.now_stock[9] += self.now_stock[5] * 506 * 0.901
                self.now_stock[5] = 0
            if e == 2:
                if i < 721:
                    self.now_stock[10] += self.now_stock[6] * 400 * 0.71
                elif i < 1465:
                    self.now_stock[10] += self.now_stock[6] * 400 * 0.742
                else:
                    self.now_stock[10] += self.now_stock[6] * 400 * 0.759
                self.now_stock[6] = 0
            if e == 3:
                if i < 721:
                    self.now_stock[11] += self.now_stock[7] * 400 * 0.70
                elif i < 1465:
                    self.now_stock[11] += self.now_stock[7] * 400 * 0.732
                else:
                    self.now_stock[11] += self.now_stock[7] * 400 * 0.749
                self.now_stock[7] = 0

        for i in range(2184):
            if self.df[i, 0] != "PROCESS":
                line_A = int(self.df[i, 0][-1]) - 1
            else:
                self.now_stock[4:8] += self.MOL_queue_A[0]
                self.MOL_queue_A[-1][line_A] = float(self.df[i, 1]) * 0.975
                self.MOL_queue_A[:-1] = self.MOL_queue_A[1:]
                self.MOL_queue_A[-1] = np.zeros([4])

            if self.df[i, 2] != "PROCESS":
                line_B = int(self.df[i, 2][-1]) - 1
            else:
                self.now_stock[4:8] += self.MOL_queue_B[0]
                self.MOL_queue_B[-1][line_B] = float(self.df[i, 3]) * 0.975
                self.MOL_queue_B[:-1] = self.MOL_queue_B[1:]
                self.MOL_queue_B[-1] = np.zeros([4])
            
            if i % 24 == 18:
                now_order = self.order[i//24]
                for e in range(4):
                    if now_order[e] != 0:
                        calc(e, now_order[e])

        p = 0
        m = 0   
        for e in self.blk_diffs:
            if e > 0:
                p += abs(e)
            else:
                m += abs(e)
        
        return p*0.2 + m*0.5
            
