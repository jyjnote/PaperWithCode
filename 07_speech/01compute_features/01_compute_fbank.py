# -*- coding: utf-8 -*-

#
# 메르 필터 뱅크 특징을 계산하는 코드입니다.
#

# wav 데이터를 읽기 위한 모듈(wave)을 임포트
import wave

# 수치 연산용 모듈(numpy)을 임포트
import numpy as np

# os, sys 모듈을 임포트
import os
import sys

class FeatureExtractor():
    ''' 특징량(FBANK, MFCC)을 추출하는 클래스
    sample_frequency: 입력 파형의 샘플링 주파수 [Hz]
    frame_length: 프레임 크기 [밀리초]
    frame_shift: 분석 간격(프레임 시프트) [밀리초]
    num_mel_bins: 메르 필터 뱅크의 수(=FBANK 특징의 차원 수)
    num_ceps: MFCC 특징의 차원 수(0차원 포함)
    lifter_coef: 리프터링 처리의 파라미터
    low_frequency: 저주파 대역 제거의 컷오프 주파수 [Hz]
    high_frequency: 고주파 대역 제거의 컷오프 주파수 [Hz]
    dither: 디더링 처리의 파라미터(잡음의 강도)
    '''
    # 클래스가 호출될 때 처음 실행되는 함수
    def __init__(self, 
                 sample_frequency=16000, 
                 frame_length=25, 
                 frame_shift=10, 
                 num_mel_bins=23, 
                 num_ceps=13, 
                 lifter_coef=22, 
                 low_frequency=20, 
                 high_frequency=8000, 
                 dither=1.0):
        # 샘플링 주파수[Hz]
        self.sample_freq = sample_frequency
        # 윈도우 크기를 밀리초에서 샘플 수로 변환
        self.frame_size = int(sample_frequency * frame_length * 0.001)
        # 프레임 시프트를 밀리초에서 샘플 수로 변환
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        # 메르 필터 뱅크의 수
        self.num_mel_bins = num_mel_bins
        # MFCC의 차원 수(0차원 포함)
        self.num_ceps = num_ceps
        # 리프터링의 파라미터
        self.lifter_coef = lifter_coef
        # 저주파 대역 제거의 컷오프 주파수[Hz]
        self.low_frequency = low_frequency
        # 고주파 대역 제거의 컷오프 주파수[Hz]
        self.high_frequency = high_frequency
        # 디더링 계수
        self.dither_coef = dither

        # FFT의 포인트 수 = 윈도우 크기 이상의 2의 거듭제곱
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # 메르 필터 뱅크 생성
        self.mel_filter_bank = self.MakeMelFilterBank()

        # 이산 코사인 변환(DCT)의 기저 행렬 생성
        self.dct_matrix = self.MakeDCTMatrix()

        # 리프터(lifter) 생성
        self.lifter = self.MakeLifter()


    def Herz2Mel(self, herz):
        ''' 주파수를 헤르츠에서 메르로 변환하는 함수
        '''
        return (1127.0 * np.log(1.0 + herz / 700))


    def MakeMelFilterBank(self):
        ''' 메르 필터 뱅크 생성 함수
        '''
        # 메르 축에서의 최대 주파수
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        # 메르 축에서의 최소 주파수
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        # 최소부터 최대 주파수까지,
        # 메르 축 상에서 균등한 간격의 주파수를 얻음
        mel_points = np.linspace(mel_low_freq, 
                                 mel_high_freq, 
                                 self.num_mel_bins+2)

        # 파워 스펙트럼의 차원 수 = FFT 크기/2 + 1
        # ※Kaldi 구현에서는 나이퀴스트 주파수 성분(마지막 +1)을 버리지만,
        # 본 구현에서는 이를 버리지 않고 사용
        dim_spectrum = int(self.fft_size / 2) + 1

        # 메르 필터 뱅크(필터 수 x 스펙트럼 차원 수)
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            # 삼각형 필터의 왼쪽, 중앙, 오른쪽 메르 주파수
            left_mel = mel_points[m]
            center_mel = mel_points[m+1]
            right_mel = mel_points[m+2]
            # 파워 스펙트럼의 각 빈에 대해 가중치를 계산
            for n in range(dim_spectrum):
                # 각 빈에 대응하는 헤르츠 축 주파수 계산
                freq = 1.0 * n * self.sample_freq / 2 / dim_spectrum
                # 메르 주파수로 변환
                mel = self.Herz2Mel(freq)
                # 해당 빈이 삼각형 필터의 범위에 있으면, 가중치를 계산
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    mel_filter_bank[m][n] = weight
         
        return mel_filter_bank

    
    def ExtractWindow(self, waveform, start_index, num_samples):
        '''
        1프레임의 파형 데이터를 추출하고, 전처리를 수행.
        또한, 로그 파워 값도 계산함.
        '''
        # waveform에서 1프레임의 파형을 추출
        window = waveform[start_index:start_index + self.frame_size].copy()

        # 디더링 처리
        # (-dither_coef~dither_coef 범위의 균등 분포 난수를 더함)
        if self.dither_coef > 0:
            window = window \
                     + np.random.rand(self.frame_size) \
                     * (2 * self.dither_coef) - self.dither_coef

        # 직류 성분 제거
        window = window - np.mean(window)

        # 후속 처리를 수행하기 전에 파워를 계산
        power = np.sum(window ** 2)
        # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
        if power < 1E-10:
            power = 1E-10
        # 로그를 계산
        log_power = np.log(power)

        # 프리엠퍼시스(고역 강조) 
        # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window, np.array([1.0, -0.97]), mode='same')
        # numpy의 합성곱에서는 0번째 요소가 처리되지 않음
        # (window[i-1]이 없으므로) 때문에,
        # window[0-1]을 window[0]으로 대체하여 처리
        window[0] -= 0.97 * window[0]

        # 해밍 윈도우 적용
        # hamming[i] = 0.54 - 0.46 * np.cos(2*np.pi*i / (self.frame_size - 1))
        window *= np.hamming(self.frame_size)

        return window, log_power


    def ComputeFBANK(self, waveform):
        ''' 메르 필터 뱅크 특징(FBANK)을 계산하는 함수
        출력1: fbank_features: 메르 필터 뱅크 특징
        출력2: log_power: 로그 파워 값(MFCC 추출 시 사용)
        '''
        # 파형 데이터의 총 샘플 수
        num_samples = np.size(waveform)
        # 특징량의 총 프레임 수 계산
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        # 메르 필터 뱅크 특징
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        # 로그 파워(MFCC 특징을 구할 때 사용)
        log_power = np.zeros(num_frames)

        # 1프레임씩 특징량을 계산
        for frame in range(num_frames):
            # 분석 시작 위치는 프레임 번호(0부터 시작) * 프레임 시프트
            start_index = frame * self.frame_shift
            # 1프레임의 파형을 추출하고 전처리를 수행
            # 또한 로그 파워 값을 얻음
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)
            
            # 빠른 푸리에 변환(FFT) 수행
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT 결과의 오른쪽 절반(음의 주파수 성분)을 제거
            # ※Kaldi 구현에서는 나이퀴스트 주파수 성분(마지막 +1)을 버리지만,
            # 본 구현에서는 이를 버리지 않음
            spectrum = spectrum[:int(self.fft_size / 2) + 1]

            # 파워 스펙트럼 계산
            spectrum = np.abs(spectrum) ** 2

            # 메르 필터 뱅크와 곱셈 후 가중치 합성
            fbank = np.dot(spectrum, self.mel_filter_bank.T)

            # 로그 계산 시 -inf가 출력되지 않도록 플로어링 처리
            fbank[fbank < 0.1] = 0.1

            # 로그를 취한 후 fbank_features에 추가
            fbank_features[frame] = np.log(fbank)

            # 로그 파워 값은 log_power에 추가
            log_power[frame] = log_pow

        return fbank_features, log_power


    def MakeDCTMatrix(self):
        ''' 이산 코사인 변환(DCT) 기저 행렬 생성
        '''
        N = self.num_mel_bins
        # DCT 기저 행렬 (기저 수(=MFCC의 차원 수) x FBANK의 차원 수)
        dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2 / N) \
                    * np.cos(((2.0 * np.arange(N) + 1) * k * np.pi) / (2 * N))

        return dct_matrix


    def MakeLifter(self):
        ''' 리프터 계산
        '''
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
        return lifter


    def ComputeMFCC(self, waveform):
        ''' MFCC를 계산하는 함수
        '''
        # FBANK와 로그 파워를 계산
        fbank, log_power = self.ComputeFBANK(waveform)
        
        # DCT 기저 행렬과 내적하여 DCT 수행
        mfcc = np.dot(fbank, self.dct_matrix.T)

        # 리프터링 수행
        mfcc *= self.lifter

        # MFCC의 0차원은 전처리 전의 파형의 로그 파워로 교체
        mfcc[:, 0] = log_power

        return mfcc


#
# 메인 함수
#
if __name__ == "__main__":

    #
    # 설정 시작
    #

    # 각 wav 파일 리스트와 특징량 출력 폴더
    train_small_wav_scp = '../data/label/train_small/wav.scp'
    train_small_out_dir = './fbank/train_small'
    train_large_wav_scp = '../data/label/train_large/wav.scp'
    train_large_out_dir = './fbank/train_large'
    dev_wav_scp = '../data/label/dev/wav.scp'
    dev_out_dir = './fbank/dev'
    test_wav_scp = '../data/label/test/wav.scp'
    test_out_dir = './fbank/test'

    # 샘플링 주파수 [Hz]
    sample_frequency = 16000
    # 프레임 길이 [밀리초]
    frame_length = 25
    # 프레임 시프트 [밀리초]
    frame_shift = 10
    # 저주파수 대역 제거의 컷오프 주파수 [Hz]
    low_frequency = 20
    # 고주파수 대역 제거의 컷오프 주파수 [Hz]
    high_frequency = sample_frequency / 2
    # 멜 필터 뱅크 특징 차원 수
    num_mel_bins = 40
    # 디저링 계수
    dither=1.0

    # 난수 시드 설정(디저링 처리 결과 재현성 보장)
    np.random.seed(seed=0)

    # 특징량 추출 클래스 호출
    feat_extractor = FeatureExtractor(
                       sample_frequency=sample_frequency, 
                       frame_length=frame_length, 
                       frame_shift=frame_shift, 
                       num_mel_bins=num_mel_bins, 
                       low_frequency=low_frequency, 
                       high_frequency=high_frequency, 
                       dither=dither)

    # wav 파일 리스트와 출력 폴더 리스트
    wav_scp_list = [train_small_wav_scp, 
                    train_large_wav_scp, 
                    dev_wav_scp, 
                    test_wav_scp]
    out_dir_list = [train_small_out_dir, 
                    train_large_out_dir, 
                    dev_out_dir, 
                    test_out_dir]

    # 각 세트에 대해 처리 실행
    for (wav_scp, out_dir) in zip(wav_scp_list, out_dir_list):
        print('입력 wav_scp: %s' % (wav_scp))
        print('출력 디렉토리: %s' % (out_dir))

        # 특징량 파일 경로, 프레임 수, 
        # 차원 수를 기록한 리스트
        feat_scp = os.path.join(out_dir, 'feats.scp')

        # 출력 디렉토리가 존재하지 않으면 생성
        os.makedirs(out_dir, exist_ok=True)

        # wav 리스트를 읽고, 
        # 특징량 리스트를 쓰기 모드로 엶
        with open(wav_scp, mode='r') as file_wav, \
                open(feat_scp, mode='w') as file_feat:
            # wav 리스트를 한 줄씩 읽기
            for line in file_wav:
                # 각 줄에는 발화 ID와 wav 파일 경로가 
                # 공백으로 구분되어 있으므로, 
                # split 함수를 사용하여 공백 구분의 줄을 
                # 리스트 형태로 변환
                parts = line.split()
                # 0번 째가 발화 ID
                utterance_id = parts[0]
                # 1번 째가 wav 파일 경로
                wav_path = parts[1]
                
                # wav 파일을 읽고, 특징량을 계산
                with wave.open(wav_path) as wav:
                    # 샘플링 주파수 확인
                    if wav.getframerate() != sample_frequency:
                        sys.stderr.write('예상 샘플링 주파수는 16000입니다.\n')
                        exit(1)
                    # wav 파일이 1채널(모노) 데이터인지 확인
                    if wav.getnchannels() != 1:
                        sys.stderr.write('이 프로그램은 모노 wav 파일만 지원합니다.\n')
                        exit(1)
                    
                    # wav 데이터의 샘플 수
                    num_samples = wav.getnframes()

                    # wav 데이터를 읽음
                    waveform = wav.readframes(num_samples)

                    # 읽은 데이터는 이진 값(16비트 정수) 
                    # 이므로 수치(정수)로 변환
                    waveform = np.frombuffer(waveform, dtype=np.int16)
                    
                    # FBANK를 계산 (log_power: 대수 파워 정보도 
                    # 출력되지만 여기서는 사용하지 않음)
                    fbank, log_power = feat_extractor.ComputeFBANK(waveform)

                # 특징량의 프레임 수와 차원 수를 구함
                (num_frames, num_dims) = np.shape(fbank)

                # 특징량 파일의 이름(splitext로 확장자 제거)
                out_file = os.path.splitext(os.path.basename(wav_path))[0]
                out_file = os.path.join(os.path.abspath(out_dir), 
                                        out_file + '.bin')

                # 데이터를 float32 형식으로 변환
                fbank = fbank.astype(np.float32)

                # 데이터를 파일에 출력
                fbank.tofile(out_file)
                # 발화 ID, 특징량 파일 경로, 프레임 수, 
                # 차원 수를 특징량 리스트에 기록
                file_feat.write("%s %s %d %d\n" % 
                    (utterance_id, out_file, num_frames, num_dims))

