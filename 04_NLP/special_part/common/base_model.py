# coding: utf-8
import sys
sys.path.append('..')
import os
import pickle
from common.np import *
from common.util import to_gpu, to_cpu


class BaseModel:
    def __init__(self):
        """
        신경망 모델의 기본 클래스입니다.

        속성:
        - params (배열의 리스트): 모델의 매개변수들입니다.
        - grads (배열의 리스트): 모델 매개변수들의 그래디언트입니다.
        """
        self.params, self.grads = None, None

    def forward(self, *args):
        """
        모델의 순전파를 수행합니다.

        Args:
        - *args: 입력 데이터를 받는 가변 인자 리스트입니다.

        Raises:
        - NotImplementedError: 이 메서드는 서브클래스에서 반드시 구현되어야 합니다.
        """
        raise NotImplementedError

    def backward(self, *args):
        """
        모델의 역전파를 수행합니다 (그래디언트 계산).

        Args:
        - *args: 입력 데이터를 받는 가변 인자 리스트입니다.

        Raises:
        - NotImplementedError: 이 메서드는 서브클래스에서 반드시 구현되어야 합니다.
        """
        raise NotImplementedError

    def save_params(self, file_name=None):
        """
        모델의 매개변수를 파일에 저장합니다.

        Args:
        - file_name (문자열 또는 None): 매개변수를 저장할 파일 이름입니다. None이면 '<클래스_이름>.pkl'로 저장됩니다.

        Notes:
        - 저장하기 전에 매개변수를 float16 타입으로 변환합니다.
        - GPU를 사용 중이라면 저장 전에 CPU로 매개변수를 이동합니다.
        """
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [to_cpu(p) for p in params]

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        """
        파일에서 모델의 매개변수를 로드합니다.

        Args:
        - file_name (문자열 또는 None): 매개변수를 로드할 파일 이름입니다. None이면 '<클래스_이름>.pkl'에서 로드합니다.

        Raises:
        - IOError: 지정된 파일이 존재하지 않을 경우 예외를 발생시킵니다.
        """
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('파일이 없습니다: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]

# 사용 예시:
# model = SomeModel()  # 모델 인스턴스화
# model.save_params('model_params.pkl')  # 모델 매개변수 저장
# model.load_params('model_params.pkl')  # 모델 매개변수 로드
