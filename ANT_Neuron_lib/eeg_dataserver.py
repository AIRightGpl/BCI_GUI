import sys

from eeg_online_stream import eego
from typing import List
from collections import deque
import time
import threading
import serial
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


global predict
predict = 0

global eeg_thread_flag
eeg_thread_flag = 1

C3 = deque(maxlen=500)
Cz = deque(maxlen=500)
C4 = deque(maxlen=500)


## create ringbuffer
class RingBuffer:

    def __init__(self, chan_lst: List[int], time_len: int, sample_rate: int =500):
        self.n_chan = len(chan_lst)
        print('Inherited channel list is {}'.format(chan_lst))
        self.n_points = time_len * sample_rate
        self.buffer = np.zeros((self.n_points, self.n_chan))
        self.overWri_flag = False
        self.currPtr = 0
        self.nextPtr = 0

    ## append buffer and update current pointer
    def append_buffer(self, data):
        n = data.shape[0]
        self.nextPtr = np.mod(self.currPtr + n, self.n_points)
        if (self.nextPtr > self.currPtr) and ((self.currPtr+n) > self.n_points):
            raise ResourceWarning("data size is greater than the size of RingBuffer, Overwrite occurs!")
            self.overWri_flag = True
        else:
            if self.overWri_flag:
                self.overWri_flag = False
            else:
                pass

        if self.overWri_flag:
            self.buffer[np.mod(np.arange(self.nextPtr, self.currPtr + n), self.n_points), :] = data
        else:
            self.buffer[np.mod(np.arange(self.currPtr, self.currPtr + n), self.n_points), :] = data
        self.currPtr = self.nextPtr

    ## get data from buffer
    def get_data(self):
        data = np.stack([self.buffer[self.currPtr:, :], self.buffer[:self.currPtr, :]])
        return data

    ## reset buffer
    def reset_buffer(self):
        self.buffer = np.zeros((self.n_points, self.n_chan))
        self.currPtr = 0
        self.nextPtr = 0


## This server is established to fill the gap between the pattern of real-time streaming and long-period data analysing
## apply RingBuffer() to store EEG data; with thread and short update_interval to stimulate real-time streaming
class DataFeedServer:

    def __init__(self, channels: List[int] = [range(32)], buff_time: int = 5, sample_rate: int = 500):
        if 'threading' not in sys.modules:
            import threading
        else:
            pass
        self._updateInterval = 0.02
        self.Ring_Buffer = RingBuffer(channels, buff_time, sample_rate=sample_rate)
        self.thread_flag = False
        self.sample_rate = sample_rate
        self.channels = channels

    ## update ringbuffer after each interval
    def stream_data_antneuron(self):
        ee = eego()
        print(eego.get_sampling_rates)
        with ee.start(self.sample_rate, 'eeg'):
            time.sleep(2)
            while self.thread_flag:
                self.thread_event.wait()
                dt = ee.get_data()
                eeg_data = np.array(dt)[:, self.channels]
                self.Ring_Buffer.append_buffer(eeg_data)
                time.sleep(self._updateInterval)

    ## apply thread to read and append to ring buffer
    def start_stream_thread(self):
        self.thread_flag = True
        self.thread = threading.Thread(target=self.stream_data_antneuron)
        self.thread_event = threading.Event()
        self.thread.start()

    ## get data from buffer
    def get_buffer_data(self):
        eegdata = self.Ring_Buffer.get_data()
        return eegdata

    ## pause streaming
    def pause_thread(self):
        self.thread_event.clear()

    ## resume streaming
    def resume_thread(self):
        self.thread_event.set()

    ## clear resource
    def unset_clear(self):
        self.thread_flag = False
        self.Ring_Buffer.reset_buffer()

if __name__ == '__main__':
    # t0 = threading.Thread(target=read_eego_data)  # eego线程
    # t0.start()
    #
    # # 流程
    # while predict == 0:
    #     pass
    # time.sleep(TIME_WAIT)
    # stand()  # 站立
    # time.sleep(TIME_WAIT)
    # for i in range(SETP_COUNT):  # 行走流程
    #     # print(predict)
    #     walk()
    #     time.sleep(TIME_WAIT)
    # reset()
    # k1.close()
    # k2.close()
    eeg_thread_flag = 0
    print(1)