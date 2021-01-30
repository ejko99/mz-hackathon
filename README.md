# mz-hackathon 🏆 웅성음성팀
## How to install
-nvidia-docker에서 작성된 파일입니다.-
```
git clone https://github.com/xodms0309/mz-hackerthon.git  
cd mz-hackerthon  
git-lfs  
pip install -r requirements.txt  
pip install .  
sudo ubuntu-drivers autoinstall
nvidia-docker build -t docker-model -f dockerfile .  
nvidia-docker run -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all docker-model  
python prediction.py --input_text test.txt --output_text result.txt
cat result.txt
```
*ckpt/734model.pth이 위치해 있는지 확인부탁드립니다:)

## About our code 💻
### for Train
- preprocessing+agmentation.ipynb : 전처리 파일
- mz_hackerton_KoBERT.ipynb : Kobert 기반 모델 학습 파일
- train.tsv : 전처리 후 train data
- test.tsv : 전처리 후 test data (dev data를 test data로 사용)

### for Validation
- dev_accuracy_결과.ipynb : dev.txt의 라벨을 제거 후 accuracy를 뽑아낸 결과

### for Test
- preprocessing.py : tsv 파일로 변환
- prediction.py : 주어진 input_text에 대해 ouput_text 출력 
- decoder.txt : decoding 위한 파일 (주어진 train.txt을 바탕으로 생성)
- ckpt/734model.pth : Kobert_base.ipynb를 기반으로 학습한 모델
 
## About our model 😎
### 1. 데이터 탐색
  * train.txt 시각화(class별 데이터의 불균형 정도와 수 등 파악)
  * train.txt 직접 확인
### 2. 전처리
  * 데이터 결측치, 중복 등 오류 제거 
  * tsv 파일로 변환
### 3. Augmentation
  * random delete
  * back translation
  * swap
### 4. KoBERT
  * 구글의 BERT 모델을 알맞게 train한 모델
  * github의 [SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT)를 baseline code로 이용
  * base line code에 layer 추가 / parameter 조정
### 5. Our Results

||train accuracy|dev accuracy|test accuracy|epochs|
|---|---|---|---|--|
|our model|93%|75.4%|72.8%|10|

### 6. reference
  [SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT)

### 그 외 시도들.. 
* lstm
* kobert2
