# -*- coding: utf-8 -*-

#
# 켑스트럼 분석을 통해 음성의
# 포르만트 성분을 추출합니다.
#

# wav 데이터를 읽기 위한 모듈(wave) 임포트
import wave

# 수치 계산을 위한 모듈(numpy) 임포트
import numpy as np

# 그래프를 그리기 위한 모듈(matplotlib) 임포트
import matplotlib.pyplot as plt

#
# 메인 함수
#
if __name__ == "__main__":
    # 열고자 하는 wav 파일
    wav_file = '../data/wav/BASIC5000_0001.wav'

    # 분석할 시간. BASIC5000_0001.wav 에서,
    # 아래 시간은 음소 "o"를 발음하고 있음
    target_time = 0.58
    # 아래 시간은 음소 "a"를 발음하고 있음
    target_time = 0.73

    # FFT(고속 푸리에 변환)를 실행할 범위의 샘플 수
    # 2의 거듭제곱이어야 함
    fft_size = 1024

    # 켑스트럼의 저차와 고차의 경계를 결정하는 차수
    cep_threshold = 33

    # 그래프를 출력할 파일(png 파일)
    out_plot = './cepstrum.png'

    # wav 파일을 열고 이후의 처리를 진행
    with wave.open(wav_file) as wav:
        # 샘플링 주파수 [Hz] 를 얻음
        sampling_frequency = wav.getframerate()

        # wav 데이터를 읽음
        waveform = wav.readframes(wav.getnframes())

        # 읽은 데이터는 바이너리 값(16비트 정수) 
        # 이므로, 수치(정수)로 변환
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # 분석할 시간을 샘플 번호로 변환
    target_index = int(target_time * sampling_frequency)

    # FFT를 수행할 구간의 파형 데이터 추출
    frame = waveform[target_index: 
                     target_index + fft_size].copy()
    
    # 해밍 윈도우를 곱함
    frame = frame * np.hamming(fft_size)

    # FFT 실행
    spectrum = np.fft.fft(frame)

    # 로그 파워 스펙트럼 계산
    log_power = 2 * np.log(np.abs(spectrum) + 1E-7)

    # 로그 파워 스펙트럼의 역 푸리에 변환을 통해
    # 켑스트럼을 계산
    cepstrum = np.fft.ifft(log_power)

    # 켑스트럼의 고차 부분을 0으로 설정
    cepstrum_low = cepstrum.copy()
    cepstrum_low[(cep_threshold+1):-(cep_threshold)] = 0.0

    # 고역 차단한 켑스트럼을 다시 푸리에 변환하여,
    # 로그 파워 스펙트럼을 계산
    log_power_ceplo = np.abs(np.fft.fft(cepstrum_low))

    # 역으로, 저차를 0으로 설정한 켑스트럼을 구함
    cepstrum_high = cepstrum - cepstrum_low
    # 단, 표시상 0차원은 0으로 하지 않음
    cepstrum_high[0] = cepstrum[0]

    # 저역 차단한 켑스트럼을 다시 푸리에 변환하여,
    # 로그 파워 스펙트럼을 계산
    log_power_cephi = np.abs(np.fft.fft(cepstrum_high))


    # 그래프 출력 영역 생성
    plt.figure(figsize=(18,10))
    
    # 로그 파워 스펙트럼의 x축(주파수 축) 생성
    freq_axis = np.arange(fft_size) \
                * sampling_frequency / fft_size
 
    # 3가지 로그 파워 스펙트럼을 그래프에 플로팅
    for n, log_pow in enumerate([log_power, 
                                 log_power_ceplo , 
                                 log_power_cephi]):
        # 그래프 영역을 3행 2열로 나누고, 첫 번째 열에 플로팅
        plt.subplot(3, 2, n*2+1)
        plt.plot(freq_axis, log_pow, color='k')

        # x축과 y축 라벨 정의
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Value')

        # 표시 영역 제한
        plt.xlim([0, sampling_frequency / 2]) 
        plt.ylim([0, 30])

    # 켑스트럼의 x축(큐프렌시 축 = 시간 축) 생성
    qefr_axis = np.arange(fft_size) / sampling_frequency

    # 3가지 켑스트럼을 그래프에 플로팅
    for n, cepst in enumerate([cepstrum, 
                               cepstrum_low , 
                               cepstrum_high]):
        # 그래프 영역을 3행 2열로 나누고, 두 번째 열에 플로팅
        plt.subplot(3, 2, n*2+2)
        # 켑스트럼은 실수부를 플로팅
        # (허수부는 거의 0에 가까움)
        plt.plot(qefr_axis, np.real(cepst), color='k')

        # x축과 y축 라벨 정의
        plt.xlabel('QueFrency [sec]')
        plt.ylabel('Value')

        # 표시 영역 제한
        plt.xlim([0, fft_size / (sampling_frequency * 2)]) 
        plt.ylim([-1.0, 2.0])

    # 그래프 저장
    plt.savefig(out_plot)
