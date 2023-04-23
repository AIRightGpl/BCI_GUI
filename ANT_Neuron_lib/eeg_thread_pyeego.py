from eeg_online_stream import eego
from collections import deque
import time
import threading
import serial
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# BCI脑电数据竞赛，左右肢体运动想象数据
data = np.load('x_BCI.npy').transpose(2, 1, 0)[:, :, 500:1000]
print(data.shape)
label = np.load('y_BCI.npy')
label = np.array([x[0] for x in label])

LDA = LinearDiscriminantAnalysis(solver='eigen')
csp = CSP(n_components=4, reg=None, log=True)
csp.fit(data[0:140], label[0:140])

train_result = csp.transform(data[0:140])
test_result = csp.transform(data[140:280])

lda = LDA.fit(train_result, label[0:140])

print(lda.predict(csp.transform(np.array([data[0]]))))

TIME_WAIT = 6  # 动作间隔时间
SETP_COUNT = 4  # 迈步次数

global predict
predict = 0

global eeg_thread_flag
eeg_thread_flag = 1

C3 = deque(maxlen=500)
Cz = deque(maxlen=500)
C4 = deque(maxlen=500)

k1 = serial.Serial()
k1.port = 'COM9'
k1.baudrate = 115200
k1.open()

k2 = serial.Serial()
k2.port = 'COM8'
k2.baudrate = 115200
k2.open()
## sdsd

def read_eego_data():
    global predict
    global eeg_thread_flag
    ee = eego()
    print(ee.get_sampling_rates)
    with ee.start(1000, 'eeg'):
        time.sleep(2)
        while eeg_thread_flag:
            dt = ee.get_data()
            eeg_data = np.array(dt)[:, 0:32]
            C3.extend(eeg_data[:, 14])
            Cz.extend(eeg_data[:, 15])
            C4.extend(eeg_data[:, 16])
            sample_data = np.array([list(C3), list(Cz), list(C4)])
            if (sample_data.shape[1] == 500):
                predict = lda.predict(csp.transform(np.array([sample_data])))[0]
            # print(predict)
            time.sleep(0.05)


def stand():
    k1.write('0'.encode('ascii'))
    k2.write('0'.encode('ascii'))
    time.sleep(0.1)

    k1.write('1'.encode('ascii'))
    k2.write('1'.encode('ascii'))
    time.sleep(3)

    k1.write('0'.encode('ascii'))
    k2.write('0'.encode('ascii'))
    print("站立")


def walk():
    k1.write('0'.encode('ascii'))
    k2.write('0'.encode('ascii'))
    time.sleep(0.1)

    k1.write('1'.encode('ascii'))
    time.sleep(3)

    k1.write('0'.encode('ascii'))
    print("迈步")


def reset():
    k2.write('0'.encode('ascii'))
    k1.write('0'.encode('ascii'))
    time.sleep(0.1)

    k2.write('1'.encode('ascii'))
    time.sleep(1)

    k2.write('0'.encode('ascii'))
    print("复位")


def sitdown():
    k1.write('0'.encode('ascii'))
    k2.write('0'.encode('ascii'))
    time.sleep(0.1)

    k2.write('1'.encode('ascii'))
    k1.write('1'.encode('ascii'))
    time.sleep(4)

    k1.write('0'.encode('ascii'))
    k2.write('0'.encode('ascii'))

    print("坐下")


## create ringbuffer
class RingBuffer():
    def __init__(self, n_chan, n_points):
        self.n_chan = n_chan
        self.n_points = n_points
        self.buffer = np.zeros((n_chan, n_points))
        self.currentPtr = 0
        self.nUpdate = 0
    ## append buffer and update current pointer
    def appendBuffer(self,data):
        n = data.shape[1]
        self.buffer[:, np.mod(np.arange(self.currentPtr, self.currentPtr+n), self.n_points)] = data
        self.currentPtr = np.mod(self.currentPtr+n-1, self.n_points) + 1
        self.nUpdate = self.nUpdate+n
    ## get data from buffer
    def getData(self):
        data = np.hstack([self.buffer[:, self.currentPtr:], self.buffer[:, :self.currentPtr]])
        return data
    # reset buffer
    def resetBuffer(self):
        self.buffer = np.zeros((self.n_chan, self.n_points))
        self.currentPtr = 0
        self.nUpdate = 0


if __name__ == '__main__':
    t0 = threading.Thread(target=read_eego_data)  # eego线程
    t0.start()

    # 流程
    while predict == 0:
        pass
    time.sleep(TIME_WAIT)
    stand()  # 站立
    time.sleep(TIME_WAIT)
    for i in range(SETP_COUNT):  # 行走流程
        # print(predict)
        walk()
        time.sleep(TIME_WAIT)
    reset()
    k1.close()
    k2.close()
    eeg_thread_flag = 0
    print(1)