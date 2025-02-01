# -*- coding: utf-8 -*-

#
# 다운로드한 라벨 데이터를 읽어와,
# 문자(한자 포함), 히라가나, 음소 단위로 정의된 라벨 파일을 생성합니다.
#

# yaml 데이터를 읽어오기 위한 모듈을 임포트
import yaml

# os 모듈을 임포트
import os

#
# 메인 함수
#
if __name__ == "__main__":
    
    # 다운로드한 라벨 데이터(yaml 형식)의 경로
    original_label = \
      '../data/original/jsut-label-master/text_kana/basic5000.yaml'

    # 라벨 데이터를 저장할 디렉토리
    out_label_dir = '../data/label/all'

    # 출력 디렉토리가 존재하지 않을 경우 생성
    os.makedirs(out_label_dir, exist_ok=True)

    # 라벨 데이터를 읽어옴
    with open(original_label, mode='r', encoding='utf-8') as yamlfile:
        label_info = yaml.safe_load(yamlfile)

    # 문자/히라가나/음소 라벨 파일을 쓰기 모드로 열기
    with open(os.path.join(out_label_dir, 'text_char'), 
              mode='w', encoding='utf-8') as label_char, \
              open(os.path.join(out_label_dir, 'text_kana'), 
              mode='w', encoding='utf-8') as label_kana, \
              open(os.path.join(out_label_dir, 'text_phone'), 
              mode='w', encoding='utf-8') as label_phone:
        # BASIC5000_0001 ~ BASIC5000_5000에 대해 반복 실행
        for i in range(5000):
            # 발화 ID
            filename = 'BASIC5000_%04d' % (i+1)
            
            # 발화 ID가 label_info에 포함되어 있지 않을 경우 에러 출력
            if not filename in label_info:
                print('Error: %s is not in %s' % (filename, original_label))
                exit()

            # 문자 라벨 정보를 가져옴
            chars = label_info[filename]['text_level2']
            # '、'와 '。'를 제거
            chars = chars.replace('、', '')
            chars = chars.replace('。', '')

            # 히라가나 라벨 정보를 가져옴
            kanas = label_info[filename]['kana_level3']
            # '、'를 제거
            kanas = kanas.replace('、', '')

            # 음소 라벨 정보를 가져옴
            phones = label_info[filename]['phone_level3']

            # 문자 라벨 파일에 1글자씩 띄어쓰기로 구분해 작성
            # (' '.join(list)는 리스트의 각 요소를 띄어쓰기로 구분하여 한 문장으로 만듦)
            label_char.write('%s %s\n' % (filename, ' '.join(chars)))

            # 히라가나 라벨 파일에 1글자씩 띄어쓰기로 구분해 작성
            label_kana.write('%s %s\n' % (filename, ' '.join(kanas)))

            # 음소 라벨은 '-'를 띄어쓰기로 변경해 작성
            label_phone.write('%s %s\n' % (filename, phones.replace('-', ' ')))
