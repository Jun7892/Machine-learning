<p align="center">
 <img src="https://item.kakaocdn.net/do/2d29be4d091e1d4d5f83c81cb1cec9414022de826f725e10df604bf1b9725cfd" width="450" height="400" />
</p>

# 7조(구황작물) 프로젝트 - Machine-Learning

### 🏠 소개

내일배움캠프에서 진행중인 이미지를 분류하는 Machine Learning 프로젝트입니다.\


### ⏲️ 개발기간

2022년 01월 12일 ~ 2022년 01월 18일

### 🧙 맴버구성 & 파트

-   Front-end : 김유경, 이형주A, 정혜원
-   Back-end : 최인호
-   AI : 김석준

### ⚙ 적용기술 및 구현 기능 적용 기술

Front-End : JavaScript, HTML&CSS\
Back-End : Flask, Python\
AI : Keras(library in Py), CNN Network\
Common : Github, Slack, Notion

### 📌 주요 기능

(1). Data Collection
- 수집대상 : 개/ 고양이
- 수집 출처 : kaggle

(2). Data Preprocessing
- 형태소 저장
- 불필요한 단어 제거

(3). Model & Algorithms
- Data argumentaion을 위해 케라스에서 제공하는 이미지 제너레이터를 사용
- 이미지의 위치를 조금 옮긴다거나, 회전, 좌우반전등을 했을 때 컴퓨터가 받아들이는 이미지는 전혀 다른것으로 인식
  이러한 변형을 줌으로써 학습 데이터를 늘리고, 이러한 변조에 강하게 모델을 학습
- target_size는 이미지를 해당 형태로 변형시켜주는데, 같은 사이즈로 데이터 읽어오기


(4). Review
- Feedback : 크롤링, 연관규칙 등 다양한 분석방법을 활용해볼 수 있어서 좋았다
- Futuer Research : rescale만 적용하여 원본 이미지 그대로 넣어서 테스트셋의 정확도와 손실율을 구해보고 그래프 분석해보기
                    



### [Notion](https://www.notion.so/37980608eb054036a945d8dfb6c0b2d6)
