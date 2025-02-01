# -*- coding: utf-8 -*-

#
# 이 코드에서는 음성 데이터와 라벨 데이터를 다운로드합니다.
# 데이터는 JSUT 코퍼스를 사용합니다.
# https://sites.google.com/site/shinnosuketakamichi/publication/jsut
#

# 파일을 다운로드하기 위한 모듈을 임포트
from urllib.request import urlretrieve

# zip 파일을 압축 해제하기 위한 모듈을 임포트
import zipfile

# os 모듈을 임포트
import os

#
# 메인 함수
#
if __name__ == "__main__":
    
    # 데이터를 저장할 디렉터리 정의
    data_dir = '../data/original'

    # 디렉터리 data_dir이 존재하지 않으면 생성
    os.makedirs(data_dir, exist_ok=True)

    # 음성 파일 (JSUT 코퍼스, zip 형식) 다운로드
    data_archive = os.path.join(data_dir, 'jsut-data.zip')
    print('JSUT 데이터 다운로드 시작')
    urlretrieve('http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip', 
                data_archive)
    print('JSUT 데이터 다운로드 완료')

    # 다운로드한 데이터를 압축 해제
    print('JSUT 데이터 압축 해제 시작')
    with zipfile.ZipFile(data_archive) as data_zip:
        data_zip.extractall(data_dir)
    print('JSUT 데이터 압축 해제 완료')

    # zip 파일 삭제
    os.remove(data_archive)

    # JSUT 코퍼스의 라벨 데이터 다운로드
    label_archive = os.path.join(data_dir, 'jsut-label.zip')
    print('JSUT 라벨 데이터 다운로드 시작')
    urlretrieve('https://github.com/sarulab-speech/jsut-label/archive/master.zip',
                label_archive)
    print('JSUT 라벨 데이터 다운로드 완료')

    # 다운로드한 데이터를 압축 해제
    print('JSUT 라벨 데이터 압축 해제 시작')
    with zipfile.ZipFile(label_archive) as label_zip:
        label_zip.extractall(data_dir)
    print('JSUT 라벨 데이터 압축 해제 완료')

    # zip 파일 삭제
    os.remove(label_archive)

    print('모든 작업 완료')
