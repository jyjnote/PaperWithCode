# -*- coding: utf-8 -*-

#
# 단시간 푸리에 변환(STFT)을 이용하여
# 음성의 스펙트로그램을 생성하는 코드
#

# wav 파일을 다루기 위한 wave 모듈 임포트
import wave

# 수치 연산을 위한 numpy 모듈 임포트
import numpy as np

# 그래프를 그리기 위한 matplotlib 모듈 임포트
import matplotlib.pyplot as plt

#
# 메인 함수
#
if __name__ == "__main__":
    # 입력할 wav 파일 경로
    wav_file = '../data/wav/BASIC5000_0001.wav'

    # 프레임 크기 (단위: 밀리초)
    frame_size = 25
    # 프레임 이동 크기 (단위: 밀리초)
    frame_shift = 10

    # 생성된 스펙트로그램을 저장할 파일 (PNG 파일)
    out_plot = './spectrogram.png'

    # wav 파일을 열고 데이터 처리 수행
    with wave.open(wav_file) as wav:
        # 샘플링 주파수 [Hz]를 가져옴
        sample_frequency = wav.getframerate()

        # wav 데이터의 총 샘플 개수를 가져옴
        num_samples = wav.getnframes()

        # wav 데이터를 읽어옴
        waveform = wav.readframes(num_samples)

        # 읽어온 데이터는 16비트 정수(binary)이므로
        # 이를 숫자(정수)로 변환
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # 프레임 크기를 밀리초에서 샘플 개수로 변환
    frame_size = int(sample_frequency * frame_size * 0.001)

    # 프레임 이동 크기를 밀리초에서 샘플 개수로 변환
    frame_shift = int(sample_frequency * frame_shift * 0.001)

    # FFT를 수행할 샘플 개수를
    # 프레임 크기 이상인 2의 거듭제곱으로 설정
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    # 단시간 푸리에 변환을 수행할 총 프레임 개수 계산
    num_frames = (num_samples - frame_size) // frame_shift + 1

    # 스펙트로그램 데이터를 저장할 행렬 생성
    spectrogram = np.zeros((num_frames, int(fft_size/2)+1))

    # 각 프레임에 대해 진폭 스펙트럼 계산
    for frame_idx in range(num_frames):
        # 분석 시작 위치 = 프레임 번호 * 프레임 이동 크기
        start_index = frame_idx * frame_shift

        # 한 프레임 길이만큼의 파형을 추출
        frame = waveform[start_index : start_index + frame_size].copy()

        # 해밍 창(Hamming window) 적용
        frame = frame * np.hamming(frame_size)

        # 빠른 푸리에 변환(FFT) 수행
        spectrum = np.fft.fft(frame, n=fft_size)

        # 진폭 스펙트럼 계산
        absolute = np.abs(spectrum)

        # FFT 결과는 좌우 대칭이므로, 왼쪽 절반만 사용
        absolute = absolute[:int(fft_size/2) + 1]

        # 로그 변환을 적용하여 로그 진폭 스펙트럼 계산
        log_absolute = np.log(absolute + 1E-7)

        # 계산 결과를 스펙트로그램 행렬에 저장
        spectrogram[frame_idx, :] = log_absolute

    #
    # 시간 영역 파형과 스펙트로그램을 플롯
    #

    # 그래프를 그릴 영역 생성
    plt.figure(figsize=(10,10))

    # 그래프를 위아래 두 개로 분할하여
    # 위쪽에 시간 영역 파형을 그림
    plt.subplot(2, 1, 1)

    # x축 (시간축) 생성
    time_axis = np.arange(num_samples) / sample_frequency

    # 시간 영역 파형을 플롯
    plt.plot(time_axis, waveform)

    # 제목 및 x축, y축 라벨 설정
    plt.title('waveform')
    plt.xlabel('Time [sec]')
    plt.ylabel('Value')

    # x축 범위를 0초부터 전체 길이까지 설정
    plt.xlim([0, num_samples / sample_frequency])

    # 아래쪽 영역에 스펙트로그램을 그림
    plt.subplot(2, 1, 2)

    # 스펙트로그램의 최대값을 0으로 맞추기 위해
    # 컬러맵의 범위를 조정
    spectrogram -= np.max(spectrogram)
    vmax = np.abs(np.min(spectrogram)) * 0.0
    vmin = - np.abs(np.min(spectrogram)) * 0.7

    # 스펙트로그램을 이미지 형태로 출력
    plt.imshow(spectrogram.T[-1::-1,:], 
               extent=[0, num_samples / sample_frequency, 
                       0, sample_frequency / 2],
               cmap = 'gray',
               vmax = vmax,
               vmin = vmin,
               aspect = 'auto')

    # 제목 및 x축, y축 라벨 설정
    plt.title('spectrogram')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')

    # 생성된 그래프를 이미지 파일로 저장
    plt.savefig(out_plot)
