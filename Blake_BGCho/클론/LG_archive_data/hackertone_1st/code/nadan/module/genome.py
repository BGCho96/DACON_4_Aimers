import os
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.special import expit

from module.new_simulator import NewSimulator
simulator = NewSimulator()

submission_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
order_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'))

class Genome():
    def __init__(self, score_ini, input_len, output_len_1, output_len_2, h1=50, h2=50, h3=50):
        # 평가 점수 초기화
        self.score = score_ini
        
        # 히든레이어 노드 개수
        self.hidden_layer1 = h1
        self.hidden_layer2 = h2
        self.hidden_layer3 = h3
        
        # Event 신경망 가중치 생성
        self.w1 = np.random.randn(input_len, self.hidden_layer1)
        self.w2 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4 = np.random.randn(self.hidden_layer3, output_len_1)
        
        # MOL 수량 신경망 가중치 생성
        self.w5 = np.random.randn(input_len, self.hidden_layer1)
        self.w6 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8 = np.random.randn(self.hidden_layer3, output_len_2)

        self.w9 = np.random.randn(input_len, self.hidden_layer1)
        self.w10 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w11 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w12 = np.random.randn(self.hidden_layer3, output_len_2)

        # Event 종류
        self.mask = np.zeros([4], np.bool) # 가능한 이벤트 검사용 마스크
        self.event_map = {0:'MODULE_1', 1:'MODULE_2', 2:"MODULE_3", 3:'PROCESS'}
        
        self.check_time = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode = 0   # 모듈 번호 1~3, stop시 0
        self.process_time = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 98

        self.line_A = 0
        self.line_B = 3
    
    def update_mask(self):
        self.mask[:] = False
        if self.process == 0:
            if self.check_time == 28:
                self.mask[:3] = True
            if self.check_time < 28:
                self.mask[self.process_mode] = True
        if self.process == 1:
            self.mask[3] = True
            if self.process_time > 98:
                self.mask[:3] = True
    
    def forward(self, inputs):
        # Event 신경망
        net = np.matmul(inputs, self.w1)
        net = self.linear(net)
        net = np.matmul(net, self.w2)
        net = self.linear(net)
        net = np.matmul(net, self.w3)
        net = expit(net)
        net = np.matmul(net, self.w4)
        net = self.softmax(net)
        net += 1
        net = net * self.mask
        out1 = self.event_map[np.argmax(net)]
        
        # MOL 수량 신경망
        net = np.matmul(inputs, self.w5)
        net = self.linear(net)
        net = np.matmul(net, self.w6)
        net = self.linear(net)
        net = np.matmul(net, self.w7)
        net = expit(net)
        net = np.matmul(net, self.w8)
        net = self.softmax(net)
        out2 = np.argmax(net)
        out2 = out2

        # MOL 수량 신경망
        net = np.matmul(inputs, self.w9)
        net = self.linear(net)
        net = np.matmul(net, self.w10)
        net = self.linear(net)
        net = np.matmul(net, self.w11)
        net = expit(net)
        net = np.matmul(net, self.w12)
        net = self.softmax(net)
        out3 = np.argmax(net)
        out3 = out3

        return out1, out2, out3

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def linear(self, x):
        return x
    
    def create_order(self, order):
        for i in range(30):
            order.loc[91+i,:] = ['0000-00-00', 0, 0, 0, 0]        
        return order

    def np_to_pd(self, np_array):
        ret = []
        for e in np_array:
            a = "PROCESS" if e[0] == 0 else f"CHECK_{int(e[0])}"
            b = e[1]
            c = "PROCESS" if e[2] == 0 else f"CHECK_{int(e[2])}"
            d = e[3]
            ret.append([a, b, c, d])
        return ret

    def get_max(self, step):
        if step < 721:
            max_value = 5.857916666666667
        elif step < 1465:
            max_value = 5.866875
        else:
            max_value = 5.87575
        return max_value
   
    def predict(self, order):
        order = self.create_order(order)
        order = np.array(order)[:, 1:]
        order = np.array(order, dtype=np.float32)

        self.submission = submission_ini
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0
        np_sub = np.zeros([2184, 4])

        stock = np.array(pd.read_csv('module/stock.csv'))[0]
        self.blk_estimate = [stock[4]*400+stock[8],
                             stock[5]*400+stock[9],
                             stock[6]*400+stock[10],
                             stock[7]*400+stock[11]]
        
        for s in range(self.submission.shape[0]):
            self.update_mask()
            
            inputs = np.array(order[s//24:(s//24+10), :]).reshape(-1) / 100000
            inputs = np.append(inputs, s/10000)
            inputs = np.append(inputs, np.array(self.blk_estimate)/10000000)

            out1, out2, out3 = self.forward(inputs)

            if out1 == 'MODULE_1':
                if self.process == 1:
                    self.process = 0
                    self.check_time = 28
                self.check_time -= 1
                self.process_mode = 0
                if self.check_time == 0:
                    self.process = 1
                    self.process_time = 0
                np_sub[s, 0] = 1
                self.line_A = 1
            elif out1 == 'MODULE_2':
                if self.process == 1:
                    self.process = 0
                    self.check_time = 28
                self.check_time -= 1
                self.process_mode = 1
                if self.check_time == 0:
                    self.process = 1
                    self.process_time = 0
                np_sub[s, 0] = 2
                self.line_A = 2
            elif out1 == 'MODULE_3':
                if self.process == 1:
                    self.process = 0
                    self.check_time = 28
                self.check_time -= 1
                self.process_mode = 2
                if self.check_time == 0:
                    self.process = 1
                    self.process_time = 0
                np_sub[s, 0] = 4
                self.line_A = 4
            elif out1 == 'PROCESS':
                self.process_time += 1
                if self.process_time == 98:
                    self.process = 0
                    self.check_time = 28
                np_sub[s, 0] = 0
            
            if 0 <= (s%168) <= 27:
                np_sub[s, 2] = 3
            else:
                np_sub[s, 2] = 0

            if np_sub[s, 0] == 0 and any(order[s//24:(s//24)+4, self.line_A-1] != 0):
                np_sub[s, 1] = out2 * self.get_max(s)
                self.blk_estimate[self.line_A-1] += out2 * 400
            else:
                np_sub[s, 1] = 0

            if np_sub[s, 2] == 0 and any(order[s//24:(s//24)+4, self.line_B-1] != 0):
                np_sub[s, 3] = out3 * self.get_max(s)
                self.blk_estimate[self.line_B-1] += out3 * 400
            else:
                np_sub[s, 3] = 0
                
        # 23일간 MOL = 0
        np_sub[:24*23+1, 1] = 0
        np_sub[:24*23+1, 3] = 0
        
        # 변수 초기화
        self.check_time = 28
        self.process = 0
        self.process_mode = 0
        self.process_time = 0

        self.submission.loc[:, "Event_A":"MOL_B"] = self.np_to_pd(np_sub)
        
        return self.submission    
    
def genome_score(genome):
    submission = genome.predict(order_ini)    
    genome.submission = submission    
    genome.score = simulator.get_score(submission)    
    return genome
