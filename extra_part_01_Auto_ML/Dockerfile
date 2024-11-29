# Python 3.8 이미지 기반으로 생성
FROM python:3.8

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    python3-dev \
    libffi-dev \
    libssl-dev

# auto-sklearn 및 Jupyter Notebook 설치
RUN pip install auto-sklearn jupyter

# 작업 디렉터리 설정
WORKDIR /workspace

# Jupyter Notebook 시작 명령어
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
