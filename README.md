# AI Fashion Recommendation Service

![Logo](https://i.imgur.com/L1fCcL9.png)

>사용자의 개별 옷 사진을 업로드하면 그 옷에 어울리는 코디들을 보여주는 프로그램입니다.
>사용자는 원하는 스타일을 선택하여 골라 추천받을 수 있습니다.
---
## Index
  - [Motivation](#Motivation)
  - [Development Environment](#Development-Environment)
  - [Key Features](#Key-features)
  - [Requirements](#Requirements)
  - [Contributor](#Contributor)
  - [Demo Video](#Demo-Video)

## Motivation

> 매년 다양한 스타일과 유행과 트렌드를 따라가기 번거로운 사람
> 
>스타일을 찾지 못한 아이템이 잉여 자원으로 낭비되고 매일 코디하기가 귀찮은 사람
>
>새로운 아이템의 매치가 궁금한 사람들을 위해 쉽게 코디하고 추천 상품까지 연결해주는 서비스를 제공하고 싶었습니다.

## Development Environment

> Pytorch
> 
> MongoDB
> 
> Flask
> 
> Streamlit
> 
> Remove background API

## Key Features

- Remove background
>Remove background API를 사용하여 사용자가 업로드한 이미지를 딥러닝 모델이 학습한 데이터에 맞추기 위해 배경을 삭제해주는 과정이 들어갑니다.
![Rembg](https://i.imgur.com/ChD28Lw.png)

- Item classifier
>사용자가 업로드한 이미지에 해당하는 상품이 무엇인지 판단합니다. efficientnet_b0모델을 사용하였고 라벨링은 옷 상품에 따라 총 21개입니다.
<div style="display:flex; justify-content:space-between;">
    <img src="https://i.imgur.com/MnwCpVJ.jpg" alt="Model" width="45%">
    <img src="https://i.imgur.com/3ep33HL.jpg" alt="Model" width="45%">
</div>

- Color classifier
>사용자가 업로드한 상품의 이미지가 어떤 색상인지 판단합니다. K-means clustering을 통해 색상 값을 추출하고 사전에 지정한 216가지 색상 값 중 가장 가까운 값을 매핑합니다.
![Color](https://i.imgur.com/RJaKWFi.png)

- Login / Sign up
>사용자 기반 서비스이므로 회원가입과 로그인 서비스를 통해 개인화된 서비스를 제공합니다.
<div style="display:flex; justify-content:space-between;">
    <img src="https://i.imgur.com/lrlGt3z.png" alt="Login" width="45%">
    <img src="https://i.imgur.com/Wo256S3.png" alt="Login" width="45%">
</div>

- Result
>AI 모델이 상품을 판단하고 해당 상품에 어울리는 옷 코디와 가격, 구매 링크 등 세부 정보까지 알려줍니다.
<div style="display:flex; justify-content:space-between;">
    <img src="https://i.imgur.com/I7WyHG5.png" alt="Login" style="width: 45%; object-fit: cover;">
    <img src="https://i.imgur.com/k5KQmo5.png" alt="Login" style="width: 45%; object-fit: cover;">
</div>

## Requirements

프로젝트를 실행하려면 먼저 필요한 라이브러리를 설치해야 합니다. 아래는 필요한 라이브러리들이 나열되어 있는 `requirements.txt` 파일을 사용하여 설치하는 방법입니다.
```sh
pip install -r requirements.txt
```

## Contributor

- 최준혁(PL)
>Github ID: acensia
>
>Database Design, Color Classifier, Data Collection, Web Crawling

- 이상윤
>Github ID: speedp001
>
>Item Classifier Model, Rembg API, Backend Design, Data Collection and Processing

- 조현준
>Github ID: Hyunjun999
>
>Database Design, Backend Design, Color Classifier, Rembg API, Data Collection

- 김동휘
>Github ID: Donghwi00
>
>Login and Sign-up, Email Verification Service, Data Collection and Processing

- 명승호
>Github ID: LuckyMspace
>
>Frontend Design, Data Collection, Color Classifier

## Demo Video

YouTube link
https://youtu.be/ONddK9AoYs4?feature=shared

PDF 자료
https://drive.google.com/file/d/1YLiWO9QMu75aqA1TJA5ot12cw2WxyeIl/view?usp=sharing

