import torch
print(torch.__version__)  # PyTorch 버전 확인
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인
print(torch.cuda.current_device())  # 현재 사용 중인 GPU 디바이스 번호 확인
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # GPU 이름 확인
