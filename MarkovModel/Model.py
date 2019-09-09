import numpy as np
import pandas as pd

_RED_, _WHITE_ = 0, 1

class Markov(object):
    def __init__(
            self,
            trans_mat=None, emis_mat=None, init_prob=None,
            observe=(), state=()
    ):
        # 状态转移概率分布
        self.trans_mat = trans_mat
        # 观测概率分布
        self.emis_mat = emis_mat
        # 初始概率分布
        self.init_prob = init_prob
        # 观测序列
        self.observe = observe
        # 状态序列
        self.state = state

    def calculateProb(self, method='forward'):
        '''
        计算观测数据的概率
        :param method: forward or backward
        :return: 观测数据的概率
        '''
        assert self.trans_mat is not None, 'transmission matrix is None'
        assert self.emis_mat is not None, 'emission matrix is None'
        assert self.init_prob is not None, 'initiation matrix is None'
        assert len(self.observe) != 0, 'observe matrix is empty'

        # 各时刻的观测概率矩阵
        # len(self.observe)个时刻
        # len(self.init_prob)个状态
        time_prob = np.ones((self.init_prob.shape[0], len(self.observe)))

        if method == 'forward':
            time_prob[:, 0] = self.emis_mat[:, self.observe[0]] * self.init_prob.T
            for t in range(1, len(self.observe)):
                time_prob[:, t] = np.dot(
                    time_prob[:, t-1], self.trans_mat
                ) * self.emis_mat[:, self.observe[t]]
            final_prob = np.sum(time_prob[:, -1], axis=0)
        else:
            for t in range(len(self.observe)-2, -1, -1):
                time_prob[:, t] = np.dot(
                    (time_prob[:, t+1] * self.emis_mat[:, self.observe[t+1]])
                    , self.trans_mat.T
                )
            final_prob = np.dot(
                (time_prob[:, 0] * self.emis_mat[:, self.observe[0]]),
                self.init_prob)
            print(time_prob)
            final_prob = np.sum(final_prob, axis=0)
        return final_prob

def main():
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5],
    ])

    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3],
    ])

    pi = np.array([
        [0.2],
        [0.4],
        [0.4],
    ])
    O = [_RED_, _WHITE_, _RED_]

    clf = Markov(A, B, pi, O)
    print(clf.calculateProb(method='forward'))
    print(clf.calculateProb(method='backward'))
if __name__ == '__main__':
    main()