# CoNLL-2003 데이터를 이용한 NER

##### 폴더 구조

```
# folder structure
BLSTM/
	CONFIG/
		POS_TAG_CONFIG.json
		WORDS_CONFIG.json
	dataset/
		test_clean.csv
		test_data.txt
		test_input.npy
		test_input_with_pos.npy
		test_labels.npy
		train_clean.csv
		train_data.txt
		train_input.npy
		train_input_with_pos.npy
		train_labels.npy
	model/
		model.h5
		model.json
		model_history.json
		model_history_with_pos.json
		model_with_pos.h5
		model_with_pos.json
	module/
		EDA.py
		preprocessing.py
		test.py
	result/
		result.txt
	EDA.ipynb
	preprocessing.py	
	README.md
	simulation.py
	simulation_with_pos.py
	test.py
	test_with_pos.py
	train.py
	train_with_pos.py
```

## 데이터의 흐름

1. 데이터는 EDA.py를 통해 txt파일에서 csv파일로 변환된다.
2. preprocessing.py에서 BLSTM 모델에 적합하게 변환되어 npy파일로 저장된다. 이 때, 코퍼스의 단어와 관련된 여러 정보들도 함께 저장된다.
3. train.py에서 npy파일로 저장된 데이터들을 훈련하고, test.py에서 모델의 성능을 측정한다.

## EDA

여러 관점에서 데이터를 분석하고, txt파일을 csv파일로 저장한다.

1. text preview
   - 텍스트를 개략적으로 살펴봄으로써 기본적으로 데이터가 어떻게 구성되어있는지, 염두해야하는 부분은 무엇인지 파악한다.
2. number of data
   - 총 데이터의 개수는 14,041개이다. 데이터 수가 많은 편은 아니므로 과적합에 유의해야한다.
3. vocab size
   - 모든 문자를 소문자로 치환하기 전 모든 단어의 개수는 23,623이다.
4. kinds of NER
   - BLSTM의 타겟값으로서 활용하는 각 단어의 NER이다. 총 개수는 10개이다.
5. hist of word count of each sentence
   - 각 문장에 포함된 단어의 개수를 살펴보면 거의 모든 문장이 70개 이하의 단어를 가지고 있다. 
6. describe word count of each sentence
   - 단어의 개수의 통계 지표를 보면 평균이 14개이고 중위값이 10개며 제 3분위수가 22개다. 따라서 input의 단어 개수를 70으로 설정하고 패딩하기로 한다.
7. log hist of count of each word
   - 각 단어의 빈도수를 log화해서 그림을 그려본 결과 각 단어들의 빈도수 차이가 많이 나는 것을 알 수 있다.
8. describe count of each word
   - 각 단어들은 평균적으로 8번 나온다고 하지만 중위수 값이 2번이다. 많은 단어들이 2번 이하로 출현한다는 것을 알 수 있다. 하지만 객체명일 경우 count가 낮을 가능성이 높으므로, min count가 일정 이상되지 않을 때, 'unkown'으로 치환하는게 합리적이다.
9. bar plot of count by NER
   - O에 해당하는 단어가 압도적으로 많다. 정확도로 결과를 측정한다면 오류를 범할 수 있다.

## Preprocessing

전처리 과정에서는 주어진 데이터들을 소문자화하고, 각 문장들을 숫자로 표현하며 70의 길이가 되도록 패딩한다.

참고 자료에 따르면 pos tag를 함께 학습한다면 좀 더 나은 결과를 얻을 수 있다고 한다. 이를 추가적으로 실험하기 위해 pos tag가 포함된 input을 추가로 만든다.

pos tag를 함께 학습하기 위해서 다음과 같은 규칙을 적용했다.

1. 모르는 단어가 

## DeepLearning

### BLSTM

### Train

### Test

### Simulation

## 실험

## 개선 방향

## 결과