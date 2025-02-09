# 🤗소개🤗(Course Introduction) <sub>-제작자:정재연-<sub>

 이 레파지토리는 인공지능을 깊이 있게 다루며, 논문 리뷰 및 코드 구현을 통해 실질적인 이해를 목표로합니다.

함께 진행하는 이 강의에서는 전통적인 딥런닝 이론부터 최신 처리 기술과 모델을 학습할 수 있습니다. 모든 코드는 <footnotesize><strong style="color:#ED5466">PyTorch</strong></footnotesize>로 제작되었으며, 실습 데이터를 통해 직접 실험할 수 있습니다. 또한, 논문과의 연관성을 강조합니다.

특히, 이 강의는 자연어 처리 분야에서 전문가가 되고자 하는 분들, 인공지능에 관심이 많고 대학원 진학을 목표로 깊이 있는 공부를 원하는 분들에게 최적화되어 있습니다. 여러분이 자연어 처리의 전문가로 성장할 수 있도록, 최신 연구와 실습을 통해 깊이 있는 학습 경험을 제공합니다.


https://github.com/labmlai/annotated_deep_learning_paper_implementations?tab=readme-ov-file
### 다양한 도구와 툴

이 강의에서는 다음과 같은 도구들과 라이브러리를 활용하여 실습을 진행할 수 있습니다:

- 
  ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)  
  컨테이너 기반의 가상 환경을 통해 일관된 개발 환경을 제공합니다.

- 
  ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)  
  실험 추적, 모델 관리 및 배포를 지원하는 오픈 소스 플랫폼입니다.

- 
  ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF4F00?style=for-the-badge&logo=Hugging%20Face&logoColor=white)  
  다양한 NLP 모델과 데이터셋을 제공하는 플랫폼입니다.

- 
  ![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)  
  데이터 파이프라인의 스케줄링 및 모니터링을 위한 플랫폼입니다.

- 
  ![Kubeflow](https://img.shields.io/badge/Kubeflow-0F4F5B?style=for-the-badge&logo=kubeflow&logoColor=white)  
  Kubernetes 환경에서 머신러닝 워크플로우를 관리하는 오픈 소스 플랫폼입니다.

- 
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)  
  머신러닝 모델을 빠르게 웹 애플리케이션으로 변환할 수 있는 라이브러리입니다.

- 
  ![Gradio](https://img.shields.io/badge/Gradio-00BFFF?style=for-the-badge&logo=gradio&logoColor=white)  
  AI 모델을 웹 인터페이스로 쉽게 배포할 수 있는 라이브러리입니다.

- 
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)  
  데이터 분석 및 시각화를 위한 대화형 노트북 환경을 제공합니다.

- 
  ![DVC](https://img.shields.io/badge/DVC-9B63C8?style=for-the-badge&logo=data%20version%20control&logoColor=white)  
  데이터와 모델 버전을 관리하고 협업을 지원하는 도구입니다.

- 
  ![Metaflow](https://img.shields.io/badge/Metaflow-5F3A65?style=for-the-badge&logo=metaflow&logoColor=white)  
  데이터 과학 작업을 관리하고 자동화하는 도구입니다.

### DevOps 및 CI/CD 도구

- 
  ![Jenkins](https://img.shields.io/badge/Jenkins-D24939?style=for-the-badge&logo=jenkins&logoColor=white)  
  오픈 소스 CI/CD 도구로, 소프트웨어 개발 파이프라인을 자동화합니다.

- 
  ![GitLab](https://img.shields.io/badge/GitLab%20CI/CD-FCA121?style=for-the-badge&logo=gitlab&logoColor=white)  
  GitLab의 통합 CI/CD 기능으로, 코드 빌드, 테스트 및 배포를 자동화합니다.

- 
  ![CircleCI](https://img.shields.io/badge/CircleCI-343434?style=for-the-badge&logo=circleci&logoColor=white)  
  클라우드 기반 CI/CD 플랫폼으로, 빠르고 효율적인 빌드 및 배포를 지원합니다.

- 
  ![Travis CI](https://img.shields.io/badge/Travis%20CI-3D3D3D?style=for-the-badge&logo=travis-ci&logoColor=white)  
  GitHub와 통합되어 빌드 및 테스트 자동화를 지원하는 CI 도구입니다.

- 
  ![Terraform](https://img.shields.io/badge/Terraform-7D5B3F?style=for-the-badge&logo=terraform&logoColor=white)  
  인프라를 코드로 정의하고 프로비저닝하는 도구입니다.


## References
<div style="border: 2px solid #ccc; padding: 10px; border-radius: 5px;">
  <h3>NLP(자연어처리)</h3>
  <table>
    <tr>
      <th>논문 주제</th>
      <th>참고문헌</th>
      <th>리뷰 및 요약</th>
      <th>코드구현</th>
      <th>참고 자료</th>
      <th>코랩 링크</th>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">Bag of Words</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1301.6770">An alternative text representation
 to TF-IDF and Bag-of-Words </a><a href="https://www.staff.city.ac.uk/~sbrp622/idfpapers/ksj_orig.pdf">A statistical interpretation of term specificity 
and its application in retrieval</a></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">TF-IDF</span></small></b></td>
      <td>
      <a href="https://arxiv.org/pdf/1301.6770">An alternative text representation
 to TF-IDF and Bag-of-Words</a>
      </td>
      <td></td>
      <td></td>
      <td><a href="https://en.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html">https://en.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html</a></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">CBOW&SG</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1301.3781">Efficient Estimation of Word Representations in
 Vector Space</a></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">Embedding</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1901.09069">Word Embeddings: A Survey</a></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">RNN</span></small></b></td>
      <td><a href="https://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf">Learning representations by back-propagation errors</a></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">LSTM</span></small></b></td>
      <td><a href="https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf">Long Short-Term Memory</a></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">seq2seq</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1409.3215">Sequence to Sequence Learning
 with Neural Networks</a></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">Mamba</span></small></b></td>
      <td></td>
      <td></td>
      <td><a href="https://github.com/alxndrTL/mamba.py">https://github.com/alxndrTL/mamba.py</a></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">Attention</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">Attention RNN</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1911.11423v2"> Single Headed Attention RNN: Stop Thinking With Your Head</a></td>
      <td></td>
      <td><a href="https://github.com/Smerity/sha-rnn/blob/218d748022dbcf32d50bbbb4d151a9b6de3f8bba/model.py#L53"> https://github.com/Smerity/sha-rnn/blob/218d748022dbcf32d50bbbb4d151a9b6de3f8bba/model.py#L53</a></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">Transformer</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1706.03762"> Attention Is All You Need</a></td>
      <td></td>
      <td></td>
      <td><a href="https://cpm0722.github.io/pytorch-implementation/transformer"> [NLP 논문 구현] pytorch로 구현하는 Transformer (Attention is All You Need)</a></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">ELMo</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">BERT</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">GPT1</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Let's build GPT: from scratch, in code, spelled out.</a></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">GPT2</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://www.youtube.com/watch?v=l8pRSuU81PU">Let's reproduce GPT-2 (124M)</a></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">ALBERT</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">RoBERTa</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">ELECTRA</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td><a href="https://colab.research.google.com/여기에_노트북_URL을_입력하세요"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" /></a></td>
    </tr>
  </table>
</div>


<hr style="border: 2px solid #ccc; margin: 10px 0;">
<div style="border: 2px solid #ccc; padding: 10px; border-radius: 5px;">
  <h3>CV(컴퓨터비전)</h3>
  <table>
    <tr>
      <th>논문 주제</th>
      <th>참고문헌</th>
      <th>리뷰 및 요약</th>
      <th>코드구현</th>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">AlexNet</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1301.6770">https://arxiv.org/pdf/1301.6770</a>, <a href="https://www.staff.city.ac.uk/~sbrp622/idfpapers/ksj_orig.pdf"></a>
      </td>
      <a href="https://drive.google.com/file/d/1XKBEXIAmFsMSHdHDSb66aVDcy7GYBz2E/view?usp=drive_link">논문 리뷰</a>
      <td></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">EfficientNet</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1301.6770">https://arxiv.org/pdf/1301.6770</a></td>
      <td></td>
      <td>
      <td><a href="https://drive.google.com/file/d/178gguYXBiLk-Py_Q2er1W2MeGIpumzPs/view?usp=drive_link">논문 리뷰</a></td>
      </td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">GoogleNet</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">RCNN</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">BGG16</span></small></b></td>
      <td><a href="https://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf">https://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf</a></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">VIT</span></small></b></td>
      <td><a href="https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf">https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf</a></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">YoloV3</span></small></b></td>
      <td><a href="https://arxiv.org/pdf/1409.3215">https://arxiv.org/pdf/1409.3215</a></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td><b><small><span style="color:#FFFF99">stable_diffusion</span></small></b></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </table>
</div>
