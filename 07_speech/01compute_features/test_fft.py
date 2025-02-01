# -*- coding: utf-8 -*-

#
# wav 파일의 일부 구간을 푸리에 변환하여
# 진폭 스펙트럼을 플롯합니다.
#

# wav 데이터를 읽어오기 위한 모듈(wave) 임포트
import wave

# 수치 계산용 모듈(numpy) 임포트
import numpy as np

# 플롯용 모듈(matplotlib) 임포트
import matplotlib.pyplot as plt

#
# 메인 함수
#
if __name__ == "__main__":
    # 열 wav 파일 경로
    wav_file = '../data/wav/BASIC5000_0001.wav'

    # 분석할 시간. BASIC5000_0001.wav에서는 
    # 아래의 시간은 음소 "o"를 발화하는 구간
    target_time = 0.58
    

    # FFT(고속 푸리에 변환)를 수행할 범위의 샘플 수
    # 반드시 2의 제곱수여야 함
    fft_size = 1024

    # 플롯을 저장할 파일(png 파일)
    out_plot = './spectrum.png'

    # wav 파일을 열고 이후 처리를 수행
    with wave.open(wav_file) as wav:
        # 샘플링 주파수 [Hz] 가져오기
        sampling_frequency = wav.getframerate()

        # wav 데이터 읽어오기
        waveform = wav.readframes(wav.getnframes())

        # 읽어온 데이터는 바이너리 값(16bit 정수)이므로
        # 숫자(정수)로 변환
        waveform = np.frombuffer(waveform, dtype=np.int16)  # dtype을 np.int16으로 수정

    # 분석할 시간을 샘플 번호로 변환
    target_index = int(target_time * sampling_frequency)

    # FFT를 수행할 구간만큼의 파형 데이터를 가져오기
    frame = waveform[target_index: target_index + fft_size]
    
    # FFT(고속 푸리에 변환) 수행
    spectrum = np.fft.fft(frame)

    # 진폭 스펙트럼 얻기
    absolute = np.abs(spectrum)

    # 진폭 스펙트럼은 좌우 대칭이므로 왼쪽 절반만 사용
    absolute = absolute[:int(fft_size / 2) + 1]

    # 로그를 취하여 로그 진폭 스펙트럼 계산
    log_absolute = np.log(absolute + 1E-7)

    #
    # 시간 파형과 로그 진폭 스펙트럼을 플롯
    #

    # 플롯을 그릴 영역 생성
    plt.figure(figsize=(10, 10))

    # 그릴 영역을 세로로 2분할하고
    # 위쪽에 시간 파형을 플롯
    plt.subplot(2, 1, 1)

    # x축(시간축) 생성
    time_axis = target_time + np.arange(fft_size) / sampling_frequency
    
    # 시간 파형 플롯
    plt.plot(time_axis, frame)

    # 플롯의 제목과 x축, y축 레이블 정의
    plt.title('Waveform')
    plt.xlabel('Time [sec]')
    plt.ylabel('Value')

    # x축 표시 영역을 분석 구간의 시간으로 제한
    plt.xlim([target_time, target_time + fft_size / sampling_frequency])

    # 분할된 플롯의 아래쪽 영역에
    # 로그 진폭 스펙트럼 플롯
    plt.subplot(2, 1, 2)

    # x축(주파수 축) 생성
    freq_axis = np.arange(int(fft_size / 2) + 1) * sampling_frequency / fft_size
    
    # 로그 진폭 스펙트럼 플롯
    plt.plot(freq_axis, log_absolute)

    # 플롯의 제목과 x축, y축 레이블 정의
    plt.title('Log-Absolute Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Log-Amplitude')

    # 플롯 저장
    plt.savefig(out_plot)

    # 플롯을 화면에 표시
    plt.show()
