# Project_LG_Welding_AD
LG전자 생산기술원 용접 비파괴 검사 시스템 이상감지 알고리즘 과제 (2023)

## rule-based feature 추출
Rule-based feature 추출은 class 형태로 작성되었습니다. scripts 폴더에 있는 rule_based.sh를 실행하면 data 폴더에 있는 모든 양극, 음극 시편에 대해 feature 추출이 가능합니다.

```
bash scripts/rule_based.sh
```

```
python extract_rule_based_features.py --filepath [FILEPATH] --output_dir [OUTPUTPATH]
```

Rule_based.ipynb 파일에 간단한 예제도 작성하였습니다.