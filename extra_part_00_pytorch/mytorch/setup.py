from setuptools import setup, find_packages

setup(
    name="torch-j",  # 패키지 이름
    version="0.1.1",            # 초기 버전
    packages=find_packages(),   # 패키지 디렉토리 자동 탐지
    install_requires=[          # 의존성 패키지들
        "numpy"                 # NumPy (기본적으로 많이 사용됨)
    ],
    long_description=open('README.md', encoding='utf-8').read(),  # 긴 설명
    long_description_content_type="text/markdown",  # 설명 형식
    url="https://github.com/jyjnote/mytorch",  # 프로젝트 URL
    author="Your Name",  # 작성자
    author_email="your_email@example.com",  # 이메일
    classifiers=[  # 패키지 분류 정보
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["deep learning pytorch minimal","jyjnote","pytorch"],  # 키워드
    python_requires='>=3.7',  # 지원하는 파이썬 버전
    project_urls={  # 추가적인 프로젝트 정보
        "Documentation": "https://github.com/jyjnote/mytorch",
        "Source": "https://github.com/jyjnote/mytorch",
    },
)
