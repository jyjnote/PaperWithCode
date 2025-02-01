# -*- coding: utf-8 -*-

#
# wav 파일을 열고 파형을 플로팅합니다.
#

# wav 데이터를 읽기 위한 모듈(wave)을 임포트
import wave

# 수치 연산용 모듈(numpy)을 임포트
import numpy as np

# 플로팅용 모듈(matplotlib)을 임포트
import matplotlib.pyplot as plt

#
# 메인 함수
#
if __name__ == "__main__":
    # 열 wav 파일
    wav_file = '../data/wav/BASIC5000_0001.wav'

    # 파형의 플로팅을 출력할 파일(png 파일)
    out_plot = './plot.png'

    # wav 파일을 열고 이후의 처리를 진행
    with wave.open(wav_file) as wav:
        # 샘플링 주파수 [Hz]를 가져옴
        sampling_frequency = wav.getframerate()

        # 샘플 크기 [Byte]를 가져옴
        sample_size = wav.getsampwidth()

        # 채널 수를 가져옴
        num_channels = wav.getnchannels()

        # wav 데이터의 샘플 수를 가져옴
        num_samples = wav.getnframes()

        # wav 데이터를 읽음
        waveform = wav.readframes(num_samples)

        # 읽어온 데이터는 바이너리 값(16bit 정수)이므로, 숫자(정수)로 변환
        waveform = np.frombuffer(waveform, dtype=np.int16)

    #
    # 읽어온 wav 파일의 정보를 출력
    #
    print("샘플링 주파수: %d [Hz]" % sampling_frequency)
    print("샘플 크기: %d [Byte]" % sample_size)
    print("채널 수: %d" % num_channels)
    print("샘플 수: %d" % num_samples)

    #
    # 읽어온 파형(waveform)을 플로팅
    #
    
    # x축(시간축)을 생성
    time_axis = np.arange(num_samples) / sampling_frequency

    # 플로팅의 그리기 영역을 생성
    plt.figure(figsize=(10,4))

    # 플로팅
    plt.plot(time_axis, waveform)

    # x축과 y축의 레이블을 정의
    plt.xlabel("시간 [초]")
    plt.ylabel("값")

    # x축의 표시 영역을 0부터 파형 종료 시점까지 제한
    plt.xlim([0, num_samples / sampling_frequency])

    # 플로팅을 저장
    plt.savefig(out_plot)
