import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fft

class SignalProcessor:

    # 筋電データ，パラメータのファイル名
    raw_signals_filename = 'raw_signals.txt'
    parameters_filemname = 'parameters.json'

    # パラメータの読み込み
    f = open(parameters_filemname)
    param = json.load(f)
    
    # サンプリング周波数
    sampling_freq = param['basic_parameters']['sampling_freq']
    # ナイキスト周波数
    nyq_freq = sampling_freq / 2
    # バタワースフィルタの次数
    butter_order = param['basic_parameters']['butter_order']
    # バンドパスフィルタの通過域端周波数
    bp_passband = np.array(param['basic_parameters']['bp_passband'])
    # ローパスフィルタのカットオフ周波数
    lp_cutoff = param['basic_parameters']['lp_cutoff']
    # FFTのサンプル数
    fft_sample_num = param['basic_parameters']['fft_sample_num']

    # 各チャネルのオフセット（フィルタ処理済み）
    offset_EMG = np.array(param['custom_parameters']['offset_EMG'])
    # 各チャネルの最大筋電量（フィルタ処理済み）
    max_EMG = np.array(param['custom_parameters']['max_EMG'])
    # 筋発揮量の閾値
    power_threshold = param['custom_parameters']['power_threshold']

    def __init__(self, title='raw_signals'):
        # 筋電データの読み込み
        self.raw_signals = np.loadtxt(SignalProcessor.raw_signals_filename, delimiter='\t', skiprows=1)
        # 筋電データのサンプル数
        self.sample_num = len(self.raw_signals)
        # 筋電データのチャネル数
        self.channel_num = len(self.raw_signals[0])
        # 筋電データの時間軸
        self.time = np.arange(0, self.sample_num / SignalProcessor.sampling_freq, 1 / SignalProcessor.sampling_freq)

        self.plot_signals(self.raw_signals, title)
        self.plot_spectrum(self.raw_signals, title)
    
    # バンドパスフィルタ
    def bandpass_filter(self, title='bp_filtered_signals'):
        self.bp_filtered_signals = np.empty(self.raw_signals.shape)
        normalized_bp_passband = SignalProcessor.bp_passband / SignalProcessor.nyq_freq
        b, a = signal.butter(SignalProcessor.butter_order, normalized_bp_passband, btype='bandpass')
        for i in range(self.channel_num):
            x = self.raw_signals[:, i]
            y = signal.filtfilt(b, a, x)
            self.bp_filtered_signals[:, i] = y

        self.plot_signals(self.bp_filtered_signals, title)
        self.plot_spectrum(self.bp_filtered_signals, title)
    
    # 全波整流
    def rectifier(self, title='rectified_signals'):
        self.rectified_signals = np.abs(self.bp_filtered_signals)
        self.plot_signals(self.rectified_signals, title)
    
    # ローパスフィルタ
    def lowpass_filter(self, title='lp_filtered_signals'):
        self.lp_filtered_signals = np.empty(self.rectified_signals.shape)
        normalized_lp_cutoff = SignalProcessor.lp_cutoff / SignalProcessor.nyq_freq
        b, a = signal.butter(SignalProcessor.butter_order, normalized_lp_cutoff, btype='lowpass')
        for i in range(self.channel_num):
            x = self.rectified_signals[:, i]
            y = signal.filtfilt(b, a, x)
            self.lp_filtered_signals[:, i] = y

        self.plot_signals(self.lp_filtered_signals, title)
        self.plot_spectrum(self.lp_filtered_signals, title)
    
    # オフセット除去
    def eliminate_offset(self, title='offset_eliminated_signals'):
        self.offset_eliminated_signals = np.empty(self.lp_filtered_signals.shape)
        for i in range(self.channel_num):
            self.offset_eliminated_signals[:, i] = self.lp_filtered_signals[:, i] - SignalProcessor.offset_EMG[i]
        # 0未満となる場合は値を0にする
        self.offset_eliminated_signals[self.offset_eliminated_signals < 0] = 0

        self.plot_signals(self.offset_eliminated_signals, title)

    # 正規化（筋発揮量の算出）
    def normalize_by_max(self, title='max_normalized_signals'):
        self.max_normalized_signals = np.empty(self.offset_eliminated_signals.shape)
        # 最大筋電量（max_EMG）が1となるように正規化
        for i in range(self.channel_num):
            self.max_normalized_signals[:, i] = self.offset_eliminated_signals[:, i] / (SignalProcessor.max_EMG[i] - SignalProcessor.offset_EMG[i])
        # 1を超える場合は値を1にする
        self.max_normalized_signals[self.max_normalized_signals > 1] = 1

        self.plot_signals(self.max_normalized_signals, title)

    # 動作有無の確認
    def check_power(self, title='signals_above_threshold'):
        self.signals_above_threshold = np.empty(self.max_normalized_signals.shape)
        # 全チャネルの筋発揮量の総和が閾値未満となる部分は安静状態とみなし，全チャネルの値を0にする
        for i in range(self.sample_num):
            sum_of_power = sum(self.max_normalized_signals[i])
            if(sum_of_power > SignalProcessor.power_threshold):
                self.signals_above_threshold[i] = self.max_normalized_signals[i]
            else:
                self.signals_above_threshold[i] = 0

        self.plot_signals(self.signals_above_threshold, title)
    
    # 正規化（筋電パターンの算出）
    def normalize_by_sum(self, title='sum_normalized_signals'):
        self.sum_normalized_signals = np.empty(self.signals_above_threshold.shape)
        # 全チャネルの総和が1となるように正規化
        for i in range(self.sample_num):
            sum_of_signals = sum(self.signals_above_threshold[i])
            if(sum_of_signals != 0):
                self.sum_normalized_signals[i] = self.signals_above_threshold[i] / sum_of_signals
            else:
                self.sum_normalized_signals[i] = 0

        self.plot_signals(self.sum_normalized_signals, title)
    
    # 波形の描画
    def plot_signals(self, signals, title):
        rows = int((self.channel_num + 3) / 2)
        columns = 2
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        fig = plt.figure(title, figsize=(columns * 6, rows * 3), tight_layout=True)
        fig.suptitle(title)
        ax_all = fig.add_subplot(rows, 1, rows, title='All Channels', xlabel='Time', ylabel='Amplitude')
        for i in range(self.channel_num):
            ax = fig.add_subplot(rows, columns, i + 1, title=f'Ch-{i + 1}', xlabel='Time', ylabel='Amplitude')
            ax.plot(self.time, signals[:, i], color=color[i])
            ax_all.plot(self.time, signals[:, i], label=f'Ch-{i + 1}')
        ax_all.legend(loc='upper right')
        plt.show()
    
    # 振幅スペクトルの描画
    def plot_spectrum(self, signals, title):
        rows = int((self.channel_num + 1) / 2)
        columns = 2
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        fig = plt.figure(title, figsize=(columns * 4, rows * 3), tight_layout=True)
        fig.suptitle(title + '_spectrum')
        freq = np.linspace(0, SignalProcessor.sampling_freq, SignalProcessor.fft_sample_num)
        for i in range(self.channel_num):
            ax = fig.add_subplot(rows, columns, i + 1, title=f'Ch-{i + 1}', xlabel='Frequency', ylabel='Amplitude')
            ax.plot(freq, np.abs(fft.fft(signals[:, i], SignalProcessor.fft_sample_num)), color=color[i])
    
def main():
    signal_processor = SignalProcessor()
    signal_processor.bandpass_filter()
    signal_processor.rectifier()
    signal_processor.lowpass_filter()
    signal_processor.eliminate_offset()
    signal_processor.normalize_by_max()
    signal_processor.check_power()
    signal_processor.normalize_by_sum()

if __name__ == "__main__":
    main()