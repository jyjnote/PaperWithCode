
#! __init__.py 파일은 파이썬에서 디렉터리를 패키지로 인식하게 합니다. 
#! 패키지 초기화와 모듈, 클래스, 함수들을 임포트하여 외부에서 쉽게 접근할 수 있도록 도와줍니다. 
#! 이를 통해 코드의 편리성과 가독성을 높일 수 있습니다.
#! 또한 패키지 로드 시 자동으로 실행되는 코드를 넣을 수도 있습니다. 
#! 패키지 내부에서 사용할 주요 기능들을 __init__.py에서 지정하여 사용자에게 필요한 기능을 간단하게 제공합니다.

#! 임포트 순서도 매우 중요함.
#! class 간 관계에 따라 순서가 초기화 되있지 않으면 에러 발생
__version__ = '0.1.0'

from mytorch.class_mytorch import Mytorch
from mytorch.class_mytorch import Function
from mytorch.class_mytorch import using_config
from mytorch.class_mytorch import no_grad
from mytorch.class_mytorch import as_array
from mytorch.class_mytorch import as_mytorch
from mytorch.class_mytorch import Parameter
from mytorch.class_mytorch  import Config
from mytorch.datasets import Dataset
from mytorch.dataloaders import DataLoader

from mytorch.layers import Layer
from mytorch.models import Model

import mytorch.optimizers
import mytorch.functions
#import mytorch.functions_conv
import mytorch.layers
import mytorch.utils

# from mytorch.class_mytorch import setup_mytorch