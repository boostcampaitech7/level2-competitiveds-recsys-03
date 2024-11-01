<h1 align="center"><a href='https://www.notion.so/110b563074a680d6a5afdb94a2b84c53?pvs=4'>RecSys-03 ㄱ해줘</a></h1>
<br></br>

## 🏆 대회 개요 🏆

  전세 시장은 매매 시장과 밀접하게 연관되어 있어, 부동산 정책과 시장 예측의 중요한 지표가 된다. 특히 전세 시장의 동향은 매매 시장과 밀접하게 연관되어 있어 부동산 정책 수립과 시장 예측에 중요한 지표로 활용된다.

- Objective : 
  **아파트의 주거 특성과 금융 지표 등 다양한 데이터를 바탕으로 전세가를 예측**

<br></br>
## 👨‍👩‍👧‍👦 팀 소개 👨‍👩‍👧‍👦
    
|강성택|김다빈|김윤경|김희수|노근서|박영균|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/TaroSin'><img src='https://github.com/user-attachments/assets/75682bd3-bcff-433e-8fe5-6515a72361d6' width='200px'/></a>|<a href='https://github.com/BinnieKim'><img src='https://github.com/user-attachments/assets/ff639e97-91c9-47e1-a0c8-a5fc09c025a6' width='200px'/></a>|<a href='https://github.com/luck-kyv'><img src='https://github.com/user-attachments/assets/015ec963-d1b4-4365-91c2-d513e94c2b8a' width='200px'/></a>|<a href='https://github.com/0k8h2s5'><img src='https://github.com/user-attachments/assets/526dc87c-0122-4829-8e94-bce6f15fc068' width='200px'/></a>|<a href='https://github.com/geunsseo'><img src='https://github.com/user-attachments/assets/0a1a27c1-4c91-4fdf-b350-1540c835ee72' width='200px'/></a>|<a href='https://github.com/0-virus'><img src='https://github.com/user-attachments/assets/98470105-260e-443d-8592-c139d7918b5e' width='200px'/></a>|

<br></br>

## 🌳 File Tree 🌳

```
{level2-competitiveds-recsys-03}
|
├──📁 EDA                        # 각자 시도한 EDA 파일
├──📁 ETC                        # 실험 파일
├──📂 data                       
|	  ├── load_dataset.py
|	  ├── merge_dataset.py          
|	  ├── data_preprocessing.py     
|	  ├── feature_engineering.py    
|	  └── README.md
|
├──📂 model                      
|	  ├── 📂 model_selection
|	  │    ├── Ensemble.py 
|	  │    └── TreeModel.py
|	  ├── data_split.py
|	  ├── feature_select.py
|	  ├── inference.py
|	  ├── model_train.py
|	  └── README.md
|
├── MLP.py                       # MLP 실행 파일
├── main.py                      # 메인 실행 파일
├── main_deposit.py              # 메인 실행 파일 (deposit_group을 활용한 deposit 예측)
├── requirements.txt             # 설치 파일
└── README.md
```

<br></br>

## ▶️ 실행 방법 ▶️

- Package install
    
    ```bash
    pip install -r requirements.txt
    ```
    
- Model training
    
    ```bash
    # 기본 실행(default = xgboost)
    python main.py
    
    # 도움말 확인
    python main.py -h
    
    # 원하는 모델 설정(xgboost, lightgbm, catboost, voting, stacking)
    python main.py --model xgboost # default: xgboost
    
    # optuna 사용 여부 설정(on/off)
    python main.py --optuna on # default: off
    
    # 프로젝트 이름 설정
    python main.py --project {project_name}
    
    # 실행 이름 설정
    python main.py --run {run_name}
    ```

<br></br>

## GitHub Convention

- ***main*** branch는 배포이력을 관리하기 위해 사용,

  ***house*** branch는 기능 개발을 위한 branch들을 병합(merge)하기 위해 사용
- 모든 기능이 추가되고 버그가 수정되어 배포 가능한 안정적인 상태라면 *house* branch에 병합(merge)
- 작업을 할 때에는 개인의 branch를 통해 작업
- EDA
    
    branch명 형식은 “**EDA-자기이름**” 으로 작성 ex) EDA-TaroSin
    
    파일명 형식은 “**name_EDA**” 으로 작성 ex) TaroSin_EDA
    
- 데이터 전처리팀 branch 관리 규칙
    
    ```
    house 
    │
    └── data
        ├── data-loader (데이터 불러오는 작업시)
        ├── data-concat (기존 데이터프레임에 새로운 feature 추가하는 작업시) 
        ├── data-feature_engineering (파생변수 생성하는 작업시)
        └── data-preprocessing (데이터 전처리 작업시)
    ```
    
- 모델팀 branch 관리 규칙
    
    ```
    house 
    │
    └── model
        ├── model-modularization (model 개발 및 모듈화 작업)
        │
        ├── model-region_cluster
        ├── model-deposit_cluster
        └── ... EDA 이름로 branch 생성
    ```
    
- *master(main)* Branch에 Pull request를 하는 것이 아닌,
    
    ***data*** Branch 또는 ***model*** Branch에 Pull request 요청
    
- commit message는 아래와 같이 구분해서 작성 (한글)

  ex) git commit -m “**docs**: {내용} 문서 작성”
  
  ex) git commit -m “**feat**: {내용} 추가”
  
  ex) git commit -m “**fix**: {내용} 수정”
  
  ex) git commit -m “**test**: {내용} 테스트”

- pull request merge 담당자 : **data - 근서** / **model - 영균** / **최종 - 성택**

  나머지는 ***house*** branch 건드리지 말 것!

  merge commit message는 아래와 같이 작성

  ex) “**merge**: {내용} 병합“
- **Issues**, **Pull request**는 Template에 맞추어 작성 (커스텀 Labels 사용)
Issues → 작업 → PR 순으로 진행

<br></br>

## Code Convention

- 문자열을 처리할 때는 큰 따옴표를 사용하도록 합니다.
- 클래스명은 `카멜케이스(CamelCase)` 로 작성합니다. </br>
  함수명, 변수명은 `스네이크케이스(snake_case)`로 작성합니다.
- 객체의 이름은 해당 객체의 기능을 잘 설명하는 것으로 정합니다.  
    ```python
    # bad
    a = ~~~
    # good
    lgbm_pred_y = ~~~
    ```
- 가독성을 위해 한 줄에 하나의 문장만 작성합니다.
- 들여쓰기는 4 Space 대신 Tab을 사용합시다.
- 주석은 설명하려는 구문에 맞춰 들여쓰기, 코드 위에 작성 합니다.
    ```python
    # good
    def some_function():
      ...
    
      # statement에 관한 주석
      statements
    ```
    
- 대구분 주석은 ###으로 한 줄 위에 작성 합니다.
    
    ```python
    # good
    ### normalization
    
    def standardize_feature(
    ```
    
- 키워드 인수를 나타낼 때나 주석이 없는 함수 매개변수의 기본값을 나타낼 때 기호 주위에 공백을 사용하지 마세요.
    
    ```python
    # bad
    def complex(real, imag = 0.0):
        return magic(r = real, i = imag)
    # good
    def complex(real, imag=0.0):
        return magic(r=real, i=imag)
    ```
    
- 연산자 사이에는 공백을 추가하여 가독성을 높입니다.
    
    ```python
    a+b+c+d # bad
    a + b + c + d # good
    ```
    
- 콤마(,) 다음에 값이 올 경우 공백을 추가하여 가독성을 높입니다.
    
    ```python
    arr = [1,2,3,4] # bad
    arr = [1, 2, 3, 4] # good
    ```
