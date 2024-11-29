# coding: utf-8
from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import sigmoid


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h


    def reset_state(self):
        self.h = None


class LSTM:
    def __init__(self, Wx, Wh, b):
        '''

        Parameters
        ----------
        Wx: 입력 x에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)
        Wh: 은닉 상태 h에 대한 가장추 매개변수(4개분의 가중치가 담겨 있음)
        b: 편향（4개분의 편향이 담겨 있음）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    """
    시간 축을 따라 LSTM을 적용하는 TimeLSTM 계층. 
    다수의 시퀀스 데이터를 처리하며, 순전파 및 역전파를 수행할 수 있다.
    
    Attributes
    ----------
    params : list
        LSTM 계층의 가중치와 편향을 포함한 리스트
    grads : list
        각 가중치에 대한 기울기를 저장하는 리스트
    layers : list
        각 타임스텝마다의 LSTM 계층을 저장
    h : numpy.ndarray
        은닉 상태를 저장
    c : numpy.ndarray
        셀 상태를 저장
    dh : numpy.ndarray
        은닉 상태의 기울기
    stateful : bool
        LSTM의 상태를 유지할지 여부를 나타내는 플래그
    
    Methods
    -------
    forward(xs)
        시퀀스 데이터를 받아 순전파를 수행하여 은닉 상태를 반환
    backward(dhs)
        시퀀스에 대한 은닉 상태의 기울기를 받아 역전파 수행
    set_state(h, c=None)
        은닉 상태와 셀 상태를 설정
    reset_state()
        은닉 상태와 셀 상태를 초기화
    """
    
    def __init__(self, Wx, Wh, b, stateful=False):
        """
        TimeLSTM 클래스 초기화 메서드.

        Parameters
        ----------
        Wx : numpy.ndarray
            입력 데이터에 대한 가중치 (4개 분할된 가중치)
        Wh : numpy.ndarray
            은닉 상태에 대한 가중치 (4개 분할된 가중치)
        b : numpy.ndarray
            편향 (4개 분할된 편향)
        stateful : bool, optional
            LSTM의 은닉 상태를 유지할지 여부 (기본값은 False)

        예시:
        >>> Wx = np.random.randn(10, 40)
        >>> Wh = np.random.randn(40, 40)
        >>> b = np.random.randn(40)
        >>> timelstm = TimeLSTM(Wx, Wh, b, stateful=True)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        """
        순전파 메서드. 시퀀스 데이터를 받아 LSTM 순전파를 수행.

        Parameters
        ----------
        xs : numpy.ndarray
            입력 데이터 (N, T, D) 형태로 주어진 시퀀스 데이터

        Returns
        -------
        hs : numpy.ndarray
            은닉 상태의 시퀀스 출력 (N, T, H) 형태

        예시:
        >>> xs = np.random.randn(2, 5, 10)  # (배치 크기 N, 시간 T, 입력 차원 D)
        >>> hs = timelstm.forward(xs)  # 순전파 수행
        >>> print(hs.shape)  # (2, 5, 40) 형태의 출력
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        """
        역전파 메서드. 시퀀스 데이터를 받아 LSTM 역전파를 수행.

        Parameters
        ----------
        dhs : numpy.ndarray
            은닉 상태의 기울기 시퀀스 (N, T, H) 형태

        Returns
        -------
        dxs : numpy.ndarray
            입력 데이터에 대한 기울기 (N, T, D) 형태

        예시:
        >>> dhs = np.random.randn(2, 5, 40)  # 은닉 상태의 기울기
        >>> dxs = timelstm.backward(dhs)  # 역전파 수행
        >>> print(dxs.shape)  # (2, 5, 10) 형태의 입력 기울기 출력
        """
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        """
        LSTM의 은닉 상태와 셀 상태를 설정하는 메서드.

        Parameters
        ----------
        h : numpy.ndarray
            은닉 상태 (N, H) 형태
        c : numpy.ndarray, optional
            셀 상태 (N, H) 형태 (기본값은 None)

        예시:
        >>> h = np.random.randn(2, 40)
        >>> c = np.random.randn(2, 40)
        >>> timelstm.set_state(h, c)
        """
        self.h, self.c = h, c

    def reset_state(self):
        """
        LSTM의 은닉 상태와 셀 상태를 초기화하는 메서드.

        예시:
        >>> timelstm.reset_state()
        """
        self.h, self.c = None, None


    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


class TimeEmbedding:
    def __init__(self, W):
        """
        TimeEmbedding 클래스 초기화 메서드.

        :param W: 임베딩 가중치 행렬 (V x D)

        예시:
        >>> W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 임베딩 가중치 행렬 초기화
        >>> time_embedding = TimeEmbedding(W)  # TimeEmbedding 객체 생성
        """
        self.params = [W]  # 가중치 행렬을 params 리스트에 저장
        self.grads = [np.zeros_like(W)]  # 가중치의 그래디언트를 0으로 초기화하여 grads 리스트에 저장
        self.layers = None  # 각 시간 스텝의 Embedding 레이어를 저장할 변수 초기화
        self.W = W  # 임베딩 가중치 저장

    def forward(self, xs):
        """
        순전파 메서드 (forward propagation).

        :param xs: 입력 데이터 (N x T) 형태, N: 배치 크기, T: 시간 스텝 수
        :return: 임베딩된 출력 (N x T x D)

        예시:
        >>> xs = np.array([[0, 2], [1, 0]])  # 입력 데이터 (2, 2)
        >>> out = time_embedding.forward(xs)  # 순전파 계산
        >>> print(out)  # 임베딩된 출력 확인
        [[0.1 0.2]
         [0.5 0.6]]
        [[0.3 0.4]
         [0.1 0.2]]
        """
        N, T = xs.shape  # N: 배치 크기, T: 시간 스텝 수
        V, D = self.W.shape  # V: 어휘 크기, D: 임베딩 차원

        out = np.empty((N, T, D), dtype='f')  # 출력 배열 초기화
        self.layers = []  # 각 시간 스텝의 Embedding 레이어를 저장할 리스트 초기화

        # 각 시간 스텝에 대해 Embedding 레이어 순전파
        for t in range(T):
            layer = Embedding(self.W)  # 현재 시간 스텝의 Embedding 레이어 생성
            out[:, t, :] = layer.forward(xs[:, t])  # 순전파 계산하여 출력에 저장
            self.layers.append(layer)  # 생성한 레이어를 리스트에 추가

        return out  # 최종 출력 반환

    def backward(self, dout):
        """
        역전파 메서드 (backward propagation).

        :param dout: 상위 레이어에서 전해받은 손실의 그래디언트 (N x T x D)
        :return: None

        예시:
        >>> dout = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])  # 손실의 그래디언트
        >>> time_embedding.backward(dout)  # 역전파 수행
        >>> print(time_embedding.grads[0])  # 가중치의 그래디언트 확인
        [[1. 0.]
         [0. 1.]
         [1. 0.]]
        """
        N, T, D = dout.shape  # dout의 형태에서 N, T, D 추출

        grad = 0  # 가중치의 그래디언트 초기화
        # 각 시간 스텝에 대해 역전파 수행
        for t in range(T):
            layer = self.layers[t]  # 현재 시간 스텝의 레이어 선택
            layer.backward(dout[:, t, :])  # 현재 레이어에 대해 역전파 수행
            grad += layer.grads[0]  # 현재 레이어의 그래디언트를 누적

        self.grads[0][...] = grad  # 누적된 그래디언트를 grads 리스트에 저장
        return None  # 반환값 없음


class TimeAffine:
    """
    시계열 데이터를 처리하는 Affine 계층 (fully-connected layer)

    Parameters
    ----------
    W : numpy.ndarray
        가중치 행렬 (D, M)
    b : numpy.ndarray
        편향 벡터 (M,)

    Attributes
    ----------
    params : list of numpy.ndarray
        가중치와 편향을 저장하는 리스트
    grads : list of numpy.ndarray
        가중치와 편향의 기울기를 저장하는 리스트
    x : numpy.ndarray
        순전파 시 입력 데이터를 저장하는 변수
    """

    def __init__(self, W, b):
        """
        TimeAffine 초기화 메서드.

        Parameters
        ----------
        W : numpy.ndarray
            가중치 행렬 (D, M)
        b : numpy.ndarray
            편향 벡터 (M,)

        예시:
        >>> W = np.random.randn(3, 4)
        >>> b = np.zeros(4)
        >>> layer = TimeAffine(W, b)
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        """
        순전파 수행.

        Parameters
        ----------
        x : numpy.ndarray
            입력 데이터 (N, T, D) (배치 크기 N, 시계열 길이 T, 입력 차원 D)

        Returns
        -------
        numpy.ndarray
            출력 데이터 (N, T, M) (배치 크기 N, 시계열 길이 T, 출력 차원 M)

        예시:
        >>> x = np.random.randn(2, 3, 3)  # 입력 데이터 (배치 크기 2, 시계열 길이 3, 입력 차원 3)
        >>> out = layer.forward(x)  # 순전파 수행
        >>> print(out.shape)  # 출력 데이터 크기 출력 (2, 3, 4)
        """
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        """
        역전파 수행.

        Parameters
        ----------
        dout : numpy.ndarray
            출력 기울기 (N, T, M)

        Returns
        -------
        numpy.ndarray
            입력 기울기 (N, T, D)

        예시:
        >>> dout = np.random.randn(2, 3, 4)  # 출력 기울기
        >>> dx = layer.backward(dout)  # 역전파 수행
        >>> print(dx.shape)  # 입력 기울기 크기 출력 (2, 3, 3)
        """
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx



class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 배치용과 시계열용을 정리(reshape)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelㅇㅔ 해당하는 데이터는 기울기를 0으로 설정

        dx = dx.reshape((N, T, V))

        return dx


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask


class TimeBiLSTM:
    def __init__(self, Wx1, Wh1, b1,
                 Wx2, Wh2, b2, stateful=False):
        self.forward_lstm = TimeLSTM(Wx1, Wh1, b1, stateful)
        self.backward_lstm = TimeLSTM(Wx2, Wh2, b2, stateful)
        self.params = self.forward_lstm.params + self.backward_lstm.params
        self.grads = self.forward_lstm.grads + self.backward_lstm.grads

    def forward(self, xs):
        o1 = self.forward_lstm.forward(xs)
        o2 = self.backward_lstm.forward(xs[:, ::-1])
        o2 = o2[:, ::-1]

        out = np.concatenate((o1, o2), axis=2)
        return out

    def backward(self, dhs):
        H = dhs.shape[2] // 2
        do1 = dhs[:, :, :H]
        do2 = dhs[:, :, H:]

        dxs1 = self.forward_lstm.backward(do1)
        do2 = do2[:, ::-1]
        dxs2 = self.backward_lstm.backward(do2)
        dxs2 = dxs2[:, ::-1]
        dxs = dxs1 + dxs2
        return dxs

# ====================================================================== #
# 이 아래의 계층들은 책에서 설명하지 않았거나
# 처리 속도보다는 쉽게 이해할 수 있도록 구현했습니다.
#
# TimeSigmoidWithLoss: 시계열 데이터용 시그모이드 + 손실 계층
# GRU: GRU 계층
# TimeGRU: 시계열 데이터용 GRU 계층
# BiTimeLSTM: 양방향 LSTM 계층
# Simple_TimeSoftmaxWithLoss：간단한 TimeSoftmaxWithLoss 계층의 구현
# Simple_TimeAffine: 간단한 TimeAffine 계층의 구현
# ====================================================================== #


class TimeSigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs, ts):
        N, T = xs.shape
        self.xs_shape = xs.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = SigmoidWithLoss()
            loss += layer.forward(xs[:, t], ts[:, t])
            self.layers.append(layer)

        return loss / T

    def backward(self, dout=1):
        N, T = self.xs_shape
        dxs = np.empty(self.xs_shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs


class GRU:
    def __init__(self, Wx, Wh):
        '''

        Parameters
        ----------
        Wx: 입력 x에 대한 가중치 매개변수(3개 분의 가중치가 담겨 있음)
        Wh: 은닉 상태 h에 대한 가중치 매개변수(3개 분의 가중치가 담겨 있음)
        '''
        self.Wx, self.Wh = Wx, Wh
        self.dWx, self.dWh = None, None
        self.cache = None

    def forward(self, x, h_prev):
        H, H3 = self.Wh.shape
        Wxz, Wxr, Wx = self.Wx[:, :H], self.Wx[:, H:2 * H], self.Wx[:, 2 * H:]
        Whz, Whr, Wh = self.Wh[:, :H], self.Wh[:, H:2 * H], self.Wh[:, 2 * H:]

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz))
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr))
        h_hat = np.tanh(np.dot(x, Wx) + np.dot(r*h_prev, Wh))
        h_next = (1-z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        H, H3 = self.Wh.shape
        Wxz, Wxr, Wx = self.Wx[:, :H], self.Wx[:, H:2 * H], self.Wx[:, 2 * H:]
        Whz, Whr, Wh = self.Wh[:, :H], self.Wh[:, H:2 * H], self.Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat =dh_next * z
        dh_prev = dh_next * (1-z)

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dWh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1-z)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # rest gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1-r)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWx))
        self.dWh = np.hstack((dWhz, dWhr, dWh))

        return dx, dh_prev


class TimeGRU:
    def __init__(self, Wx, Wh, stateful=False):
        self.Wx, self.Wh = Wx, Wh
        selfdWx, self.dWh = None, None
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        N, T, D = xs.shape
        H, H3 = self.Wh.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = GRU(self.Wx, self.Wh)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        N, T, H = dhs.shape
        D = self.Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        self.dWx, self.dWh = 0, 0

        dh = 0
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)

            dxs[:, t, :] = dx
            self.dWx += layer.dWx
            self.dWh += layer.dWh

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class Simple_TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        layers = []
        loss = 0

        for t in range(T):
            layer = SoftmaxWithLoss()
            loss += layer.forward(xs[:, t, :], ts[:, t])
            layers.append(layer)
        loss /= T

        self.cache = (layers, xs)
        return loss

    def backward(self, dout=1):
        layers, xs = self.cache
        N, T, V = xs.shape
        dxs = np.empty(xs.shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs


class Simple_TimeAffine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.dW, self.db = None, None
        self.layers = None

    def forward(self, xs):
        N, T, D = xs.shape
        D, M = self.W.shape

        self.layers = []
        out = np.empty((N, T, M), dtype='f')
        for t in range(T):
            layer = Affine(self.W, self.b)
            out[:, t, :] = layer.forward(xs[:, t, :])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, M = dout.shape
        D, M = self.W.shape

        dxs = np.empty((N, T, D), dtype='f')
        self.dW, self.db = 0, 0
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout[:, t, :])

            self.dW += layer.dW
            self.db += layer.db

        return dxs




