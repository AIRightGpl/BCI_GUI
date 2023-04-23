import numpy as np
import wraped_eego as we

EEG_Names = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1",  ##13
    "T7", "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",  ##27
    "P8", "POz", "O1", "Oz", "O2"
]  # 32通道

BIP_Names = ['BIP' + "%02d" % i for i in range(24)]  # 24导（6*4）双极性输入，包括EMG
Extra_Names = ["Trigger", 'Counter']

EEG_Channel_Names = EEG_Names + BIP_Names + Extra_Names  # 测信号，58通道
Impedance_Channel_Names = EEG_Names + Extra_Names  # 测阻抗，34通道


SampleFreq = 250  ##脑电信号的采样频率
bandpass_fmin = 2  ##带通滤波，下限频率  ## 8-26Hz
bandpass_fmax = 80  ##带通滤波，上限频率
filte_order = 10  ##fliter_order阶巴特沃斯滤波器
notch_freq = 50  ##陷波滤波器，去除工频（50Hz）干扰
quality_factor = 50  ##陷波滤波器的品质因数 Dimensionless parameter that characterizes notch filter -3 dB bandwidth relative to its center frequency, .bwQ = w0/bw
pick_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'FC5', 'CP5', 'CP1', 'CP6', 'P7', 'P4', 'P3', 'POz']
pick_cha_num = [0, 2, 4, 6, 3, 7, 8, 19, 20, 22, 23, 26, 24, 28]
##选择脑电的通道，通道数与通道名对应为：
##ANT-neuron-32-channal:{ Fp1--1;Fpz--2;Fp2--3;P7--4;F3--5;Fz--6;F4--7;F8--8;FC5--9;FC1--10;FC2--11;FC3--12;
##  M1--13;T7--14;C3--15;Cz--16;C4--17;T8--18;M2--19;CP5--20;CP1--21;CP2--22;CP6--23;P7--24;P3--25;Pz--26;
##  P4--27;P8--28;POz--29;O1--30;Oz--31;O2--32 }

from scipy import signal

def creat_band_pass(flitorder,f_min,f_max,sampfreq):
    nyq = 0.5 * sampfreq
    low = f_min / nyq
    high = f_max / nyq
    sos = signal.butter(flitorder, [low, high], btype='band', output='sos')
    return sos

def band_pass(data,filt):
    ##band, asip = signal.butter(flitorder, [2.0 * f_min / sampfreq, 2.0 * f_max / sampfreq], 'bandpass')
    ##data = signal.filtfilt(band, asip, data.T)
    ##data = data.T ##以上代码运行后产生的数据是全为NaN的，改成一个高通一个低通后：产生的数据数量级为e^26显然有问题
    ##网上大量的滤波例子不适用于EEG信号滤波，因为：通带相较于采样频率过于狭窄，同时scipy在产生高阶滤波器时有bug，在任何平台上，-
    ##高阶滤波器都没有办法通过单级实现，最新更新的scipy可以产生sos，使用sosfiltfilt()
    ##详细内容见：https://dsp.stackexchange.com/questions/17235/filtfilt-giving-unexpected-results?rq=1
    dat_done = signal.sosfiltfilt(filt, data.T)
    return dat_done.T

def creat_notch_gridfreq(centerfreq, qfactor, sampfreq):
    w0 = centerfreq/(sampfreq/2)  ## in v-0.19.1 the function require a normalized notch frequency
    # ba, ap = signal.iirnotch(centerfreq, qfactor, sampfreq)  ## that is useage in v-1.x.x or higher
    ba, ap = signal.iirnotch(w0, qfactor)
    return ba, ap

def notch_filte(data,b,a):
    d_not = signal.filtfilt(b, a, data.T)
    return d_not.T

def re_inference(data, inferencechan):
    (lenth, wide) = data.shape
    chalst = list(range(wide))
    del chalst[inferencechan]
    for i in chalst:
        data[:, i] = data[:, i] - data[:, inferencechan]
    return data


class eego():
    def __init__(self):
        self.weego = we.eego()
        self.sampling_rates = None
        self.started = False
        self.channel_names = None

    @property
    def get_sampling_rates(self):
        if self.sampling_rates is None:
            self.sampling_rates = self.weego.getSamplingRatesAvailable()  # 获取可设置的采样率
        return self.sampling_rates

    def start(self, sampling_rate='min', stream_type='eeg'):

        self.weego.connect_amplifier()

        if stream_type == 'impedance':
            self.weego.openImpedanceStream()
            self.started = True
            self.channel_names = Impedance_Channel_Names
        elif stream_type == 'eeg':
            if sampling_rate == 'min':
                sampling_rate = self.get_sampling_rates[0]
                print(f"Available Sampling Rates: {self.sampling_rates}")
                print(f'Set to {sampling_rate} Hz.')
            elif sampling_rate == 'max':
                sampling_rate = self.get_sampling_rates[-1]
                print(f"Available Sampling Rates: {self.sampling_rates}")
                print(f'Set to {sampling_rate} Hz.')
            try:
                self.weego.openEegStream(sampling_rate)
            except RuntimeError as e:
                print(f"Available Sampling Rates: {self.sampling_rates}")
                raise e
            self.started = True
            self.channel_names = EEG_Channel_Names
        else:
            raise ValueError('Invalid stream_type %s' % stream_type)
        return self

    def get_data(self):
        return self.weego.getData()

    def release(self):
        self.weego.release()
        self.sampling_rates = None
        self.started = False
        self.channel_names = None

    def __enter__(self):
        if not self.started:
            self.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()
        return False

if __name__ == '__main__':
    ee = eego()
    with ee.start(1000,'eeg'):
        data = ee.get_data()
    print('done')