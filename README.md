# Sentimental_Analysis (Pretrained-bert-model)

   - 네이버 영화평을 직접 크롤링하여 감정분석 진행


           datasets
           ├── bert_classification.ipynb
           ├── best_train.csv
           └── worst_train.csv

## Prerequisites

- Python 3.6+
- Tensorflow
- Sklearn
- Transformer_huggingface

## Process

### 1. 개요


- 프로젝트 후 RNN, LSTM 모델로 분석 진행하였지만, 효과 미미

    NLP 논문 스터디 및 구현을 진행<br>
    Bert 모델이 감정분석에 우수한 결과를 도출함

- Bert 모델 공부 (직접 데이터 pretraining , pretrained 모델 사용)


### 2. Data collection & Data Preprocessing & Modeling

- 네이버 영화평이 많은 Avengers 영화평 수집

① 영화평 : Document , 별점 : label<br>

- 네이버 영화평의 특성 상, 사람들마다 평점의 기준이 다름<br>별점 7점 미만일 때 부정적 리뷰가 굉장히 많음을 확인

        - 별점 7점 이상 : 1 (긍정)
        - 별점 7점 미만 : 0 (부정)<br>


② 데이터 수집 시, training set의 target attribute 개수 조정<br>

- 데이터의 target attribute의 클래스 불균형 해소

        - Training set = 별점 하위 기준 10000개, 별점 상위 기준 10000개
        - Test set = 10000개 (class 0 : 470, class 1 : 9530)

③ huggingface 패키지 사용<br>

        - BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        - SEQ_LEN 데이터의 최댓값 부여


④ Pretrained model 사용 <br>

        - TFBertModel.from_pretrained('bert-base-multilingual-cased')
        - 출력층에 Dense(1) 추가, activation = ‘sigmoid’ 함수

### 3. Evaluation

- Test set : tokenizer 후, predict 함수로 예측
  - 확률을 기준으로 0.5 이상 : 1 (긍정) , 0.5 미만 : 0 (부정) 할당
  - Accuracy, Precision, Recall 등 기본 performance 비교

### 4. 시사점 및 개선 방향

- 별점 조정, 클래스 불균형 해소, finetuning 을 통하여 성능 개선 달성

- 향후 한국어 맞춤의 Kobert 모델 도입 시, 성능 개선 기대됨
- 영화평 전용의 데이터 수집을 많이 하여, 이를 직접 pretraining 하여
  일명 movibert모델을 생성할 계획

## References

- https://github.com/kimwoonggon/publicservant_AI/blob/master/1_(HuggingFace%2BTF2)%EB%84%A4%EC%9D%B4%EB%B2%84_%EC%98%81%ED%99%94_%ED%8F%89%EA%B0%80_%EA%B8%8D%EB%B6%80%EC%A0%95_%EB%B6%84%EC%84%9D.ipynb


## Author

HyoJun Kim / [blog](http://rlagywns0213.github.io/)
