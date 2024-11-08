import numpy as np
from sklearn import linear_model, neural_network, svm, ensemble, decomposition, manifold, gaussian_process
from sklearn.preprocessing import MinMaxScaler as MM
import xlrd
from sklearn.random_projection import GaussianRandomProjection as GaussRP
from IGPR import IGPR
import time
import random
import os
from math import sqrt
import warnings
from sklearnex import patch_sklearn
patch_sklearn()

# 随机矩阵
def GenerateProjectionMatrix(data_dimension, d_r, flag):
    """
    This function generates different types of projection matrixs.

    Parameters:
        data_feature: Numpy array, the data to be projected
        d_r: the dimension reduced
        flag: string type(1.SPM1;2.SPM2;3.ESPM;4.CPM),the flag indicating the type of matrix you want to generate
    """
    # data_dimension=data_feature.shape[1]
    # d_r=data_dimension//10
    if flag == 'SPM1':
        # The first Sparse Projection Matrix
        rand = np.random.rand(data_dimension, d_r)
        project_matrix = np.zeros((data_dimension, d_r))
        project_matrix[rand < 1 / 6] = -np.sqrt(3) / np.sqrt(d_r)
        project_matrix[rand > 5 / 6] = np.sqrt(3) / np.sqrt(d_r)
        return project_matrix
    if flag == 'SPM2':
        # The Second Sprse Projection Matrix
        rand = np.random.rand(data_dimension, d_r)
        project_matrix = np.zeros((data_dimension, d_r))
        project_matrix[rand < 1 / 2] = -1 / np.sqrt(d_r)
        project_matrix[rand > 1 / 2] = 1 / np.sqrt(d_r)
        return project_matrix
    if flag == 'ESPM':
        # The extreme Sparse Projection Matrix
        rand = np.random.rand(data_dimension, d_r)
        project_matrix = np.zeros((data_dimension, d_r))
        project_matrix[rand < 1 / (2 * np.sqrt(d_r))] = -np.sqrt(d_r)
        project_matrix[rand > (1 - 1 / (2 * np.sqrt(d_r)))] = np.sqrt(d_r)
        return project_matrix
    if flag == 'CPM':
        # The circular Projection Matrix
        rand_value = np.random.randn(data_dimension)
        d = np.random.binomial(1, 0.5, data_dimension)
        D = np.diag(d)
        project_matrix = np.zeros((data_dimension, d_r))

        index = np.arange(data_dimension)
        for i in range(d_r):
            project_matrix[(index + i) % data_dimension, i] = rand_value
        project_matrix = np.matmul(D,project_matrix)/np.sqrt(d_r)
        return project_matrix
#############################################################################################################
#############################################################################################################

class samples_load():
    def __init__(self, fname):
        self.fname = fname

    def load_data(self, sheetname='Sheet1', start_row=0, label_exist=True):
        wb = xlrd.open_workbook(filename=self.fname)
        wl = wb.sheet_by_name(sheetname)
        if label_exist:
            end_col = wl.ncols - 1  # 文件列数-1
        else:
            end_col = wl.ncols
        low_data = np.zeros((wl.nrows - start_row, end_col))
        if label_exist: low_target = np.zeros(wl.nrows - start_row)
        for r in range(start_row, wl.nrows):
            for c in range(end_col):
                low_data[r - start_row, c] = wl.cell(r, c).value

            if label_exist: low_target[r - start_row] = (wl.cell(r, -1).value)
        if label_exist:
            return low_data, low_target
        else:
            return low_data

#############################################################################################################
#############################################################################################################


class HierarchicalMFM():
    def __init__(self,iter=10, LFM_name='GRP', weak_learner='RandomForest', k=5, HFM_name='GP'):
        self.LFM_name, self.weak_learner = LFM_name, weak_learner
        self.HFM_name = HFM_name
        self.LFM, self.HFM = None, None
        self.LF_num = k
        self.Standar = MM()
        self.iter = iter
        self.transformer = []
        self.DR = None

    def train_LF(self, lowdata, lowtarget):
        if self.LFM_name == 'GRP':
            self.LFM = []
            for i in range(self.LF_num):
                transformer = GaussRP(n_components=lowdata.shape[1], random_state=i*self.iter)
                new_lowdata = transformer.fit_transform(lowdata)
                if self.weak_learner == 'Adaboost':
                    wl_base = ensemble.RandomForestRegressor(n_estimators=100)
                    wl = ensemble.AdaBoostRegressor( n_estimators=50, base_estimator=wl_base)
                elif self.weak_learner == 'RandomForest':
                    wl = ensemble.RandomForestRegressor(n_estimators=100)
                wl.fit(new_lowdata,lowtarget)
                self.LFM.append(wl)
        else:
            self.LFM = []
            for i in range(self.LF_num):
                np.random.seed(i*self.iter)
                transformer = GenerateProjectionMatrix(lowdata.shape[1], lowdata.shape[1], self.LFM_name)
                self.transformer.append(transformer)
                new_lowdata = np.matmul(lowdata, transformer)
                if self.weak_learner == 'Adaboost':
                    wl_base = ensemble.RandomForestRegressor(n_estimators=100)
                    wl = ensemble.AdaBoostRegressor(n_estimators=50, base_estimator=wl_base)
                elif self.weak_learner == 'RandomForest':
                    wl = ensemble.RandomForestRegressor(n_estimators=50)
                wl.fit(new_lowdata, lowtarget)
                self.LFM.append(wl)


    def dc(self, data):
        n = len(data)
        y = np.zeros((n, self.LF_num))
        if self.LFM_name == 'GRP':
            for i in range(self.LF_num):
                transformer = GaussRP(n_components=data.shape[1], random_state=i*self.iter)
                new_data = transformer.fit_transform(data)
                y[:, i] = self.LFM[i].predict(new_data)
        else:
            for i in range(self.LF_num):
                np.random.seed(i*self.iter)
                #transformer = GenerateProjectionMatrix(data.shape[1], data.shape[1], self.LFM_name)
                new_data = np.matmul(data, self.transformer[i])
                y[:, i] = self.LFM[i].predict(new_data)
        return y


    def nor(self, lowdata):
        self.Standar.fit(self.dc(lowdata))


    def train_HF(self, hdata, htarget):
        if self.HFM_name == 'GP':
            a = 0
            for (x, y) in zip(self.Standar.transform(self.dc(hdata)), htarget):
                Y = []
                Y.append(y)
                Y = np.array(Y)
                if a == 0:
                    regr = IGPR(x, Y, 1, 2, 0.1)
                else:
                    regr.learn(x, Y)
                a += 1

        elif self.HFM_name == 'Lasso':
            regr = linear_model.Lasso()
            regr.fit(self.dc(hdata), htarget)
        else:
            print('没有此方法')
        self.HFM = regr
        # 对high_model训练


    def retrain_HF(self, hdata, htarget):
        if self.HFM_name == 'GP':
            for (x, y) in zip(self.Standar.transform(self.dc(hdata)), htarget):
                Y = []
                Y.append(y)
                Y = np.array(Y)
                self.HFM.learn(x, Y)
        else:
            print('此方法没有增量形式')


    def train(self, lowdata, lowtarget, highdata, hightarget):

        self.train_LF(lowdata, lowtarget)
        if self.HFM_name == 'GP' or self.HFM_name == 'ftrl' or self.HFM_name == 'SGD':
            self.nor(lowdata)
        self.train_HF(highdata, hightarget)


    def predict(self, data):
        dcdata = self.dc(data)
        newdata = dcdata
        if self.HFM_name == 'GP':
            Y = np.array([])
            for x in self.Standar.transform(newdata):
                Y = np.append(Y, self.HFM.predict(x)[0])
            return Y
        elif self.HFM_name == 'Lasso' or self.HFM_name == 'MLP':
            return self.HFM.predict(newdata)
        else:
            print('没有此种方法')


    def predict_cov(self, data):
        dcdata = self.dc(data)
        newdata = self.DR.transform(dcdata)
        if self.HFM_name == 'GP':
            Cov = np.array([])
            for x in self.Standar.transform(newdata):
                Cov = np.append(Cov, self.HFM.predict(x)[1])
            return Cov
        else:
            print('没有此种方法')

#############################################################################################################
#############################################################################################################
def choose(y):
    b = np.var(y, axis=0)
    a = list(b)
    add_max = a.index(max(a))
    return add_max

def distance(high_xcom, high_xpool):
    dis = []
    for x in high_xcom:
        dis.append(np.sqrt(np.sum(np.asarray(x - high_xpool) ** 2, axis=1)))
    return np.min(dis, axis=0)



def QBC_2(iter,l_num, K, hnum, need_hnum, ldata, ltarget, high_xfixed, high_yfixed, high_xpool, high_ypool):
    random.seed(None)
    if len(high_yfixed) >= need_hnum:
        print('高保真度样本数量充足')
        return high_yfixed.index()  # 若训练样本数量充足，返回前num个样本的序号

    add_xhigh = []
    add_yhigh = []
    high_xcom = high_xfixed
    high_ycom = high_yfixed

    # 当训练样本数量不足时
    while hnum < need_hnum:
        # 计算x_pool中所有样本的误差从大到小并排序
        c_num = 6
        y = []
        j = 0
        while j < c_num:
            a = random.sample(range(l_num), int(l_num/2))
            low_xcom = ldata[a]
            low_ycom = ltarget[a]
            hmfm_qbc = HierarchicalMFM(iter=iter, LFM_name='GRP', weak_learner='RandomForest', k=K, HFM_name='GP')
            hmfm_qbc.train(low_xcom, low_ycom, high_xcom, high_ycom)
            y.append(hmfm_qbc.predict(high_xpool))
            # print(y)

            j += 1

        info = np.var(y, axis=0)*(distance(high_xcom, high_xpool))**(1)
        ind = np.argsort(-info)
        # 根据排序结果挑选出样本满足数量要求
        max_add = ind[0]
        add_xhigh.append(list(high_xpool[max_add]))
        add_yhigh = np.append(add_yhigh, high_ypool[max_add])
        # print(add_xhigh, add_yhigh)
        high_xcom = np.append(high_xcom, [high_xpool[max_add]], axis=0)
        high_ycom = np.append(high_ycom, high_ypool[max_add])
        high_xpool = np.delete(np.array(high_xpool), max_add, axis=0)
        high_ypool = np.delete(np.array(high_ypool), max_add)
        hnum = hnum + 1

    return add_xhigh, add_yhigh, high_xcom, high_ycom, high_xpool, high_ypool
#############################################################################################################
#############################################################################################################


class AL_HRIGP_MFS():
    def __init__(self, iter, hfmf, lowdata, lowtarget, high_xfixed, high_yfixed, high_xpool, high_ypool, c_num):
        self.model = hfmf
        self.lowdata, self.lowtarget = lowdata, lowtarget
        self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool = high_xfixed, high_yfixed, high_xpool, high_ypool
        self.c_num = c_num
        self.l_num = len(self.lowtarget)
        self.h_num = len(self.high_yfixed)
        self.iter = iter

    def AL(self,everyadd_num,AL_name):
        if AL_name == 'US':
            add_xhigh, add_yhigh, self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool = US(self.l_num, self.c_num, self.h_num, self.h_num+everyadd_num, self.lowdata, self.lowtarget,
                                                                      self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool)
            self.h_num = self.h_num+everyadd_num
        elif AL_name == 'QBC':
            add_xhigh, add_yhigh, self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool = QBC(self.iter,self.l_num, self.c_num, self.h_num, self.h_num+everyadd_num, self.lowdata, self.lowtarget,
                                                                      self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool)
            self.h_num = self.h_num + everyadd_num
        elif AL_name == 'DW_QBC':
            add_xhigh, add_yhigh, self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool = QBC_2(self.iter,self.l_num, self.c_num, self.h_num, self.h_num+everyadd_num, self.lowdata, self.lowtarget,
                                                                      self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool)
            self.h_num = self.h_num + everyadd_num
        elif AL_name == 'maxcov':
            add_xhigh, add_yhigh, self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool = maxcov(self.l_num, self.c_num, self.h_num, self.h_num+everyadd_num, self.lowdata, self.lowtarget,
                                                                      self.high_xfixed, self.high_yfixed, self.high_xpool, self.high_ypool)
            self.h_num = self.h_num + everyadd_num
        elif AL_name == 'random':
            random.seed(None)
            a = random.sample(range(len(self.high_ypool)), everyadd_num)
            add_xhigh = self.high_xpool[a]
            add_yhigh = self.high_ypool[a]
            self.high_xfixed = np.append(self.high_xfixed, add_xhigh, axis=0)
            self.high_yfixed = np.append(self.high_yfixed, add_yhigh)
            self.high_xpool = np.delete(self.high_xpool,a,axis=0)
            self.high_ypool = np.delete(self.high_ypool, a, axis=0)
            self.h_num = self.h_num + everyadd_num
        else: print('没有此种主动学习方法')
        #print(add_xhigh, add_yhigh)
        return add_xhigh, add_yhigh

    def remake(self,everyadd_num,AL_name):
        add_xhigh, add_yhigh = self.AL(everyadd_num,AL_name)
        self.model.retrain_HF(np.array(add_xhigh), np.array(add_yhigh))
        return self.model

    def get_traindata(self):
        return self.high_xfixed, self.high_yfixed


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings('ignore')
    def NRMSE(y, ytest, r_num):
        nrmse = []
        for i in range(r_num):
            sum_1 = np.sum(np.power(y[i] - ytest, 2))
            sum_2 = np.sum(np.power(ytest, 2))
            nrmse.append(sqrt(sum_1 / sum_2))
        NRMSE = np.mean(nrmse)
        return NRMSE


    datanames = ['beale', 'bran', 'hart3', 'hart6', 'math8', 'f9']
    for d in datanames:
        print(d+'---------------------------------------------------------')
        dataload = samples_load('./code/' + d + '.xls')
        lowdata, lowtarget = dataload.load_data(sheetname='low', start_row=1)
        highdata, hightarget = dataload.load_data(sheetname='high', start_row=1)
        tdata, ttarget = dataload.load_data(sheetname='test', start_row=1)

        l_num = lowdata.shape[1]*100

        for h_num in [3,8,15,20]:
            #c_num = 6
            #need_hnum = 25
            N = 10
            random.seed(12345)
            y_all = []
            for i in range(N):
                #random.seed(1)
                a = random.sample(range(len(lowdata)), l_num)
                ltarget = lowtarget[a]
                ldata = lowdata[a]
                b = random.sample(range(len(highdata)), (len(highdata)))
                high_xfixed = highdata[b[0:h_num]]
                high_yfixed = hightarget[b[0:h_num]]

                MFS = HierarchicalMFM(iter=1, LFM_name='ESPM', weak_learner='Adaboost', k=5, HFM_name='GP')
                t1 = time.time()
                MFS.train(ldata, ltarget, high_xfixed, high_yfixed)
                t2 = time.time()
                y = MFS.predict(tdata)
                t3 = time.time()
                #print('h_num='+ str(h_num) + '  K='+ str(K))
                print('train-time='+str(t2-t1),'test-time='+str(t3-t2))
                y_all.append(y)
            error = NRMSE(y_all, ttarget, N)
            print('h_num=' + str(h_num) + '  K=' + str(5) + '     error=' + str(error))

#1.SPM1;2.SPM2;3.ESPM;4.CPM