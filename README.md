# atopy_predict_detection



#### 연구소 프로젝트 기록(피부 이미지를 인식하여 아토피인지 아닌지 예측하고 그 부분을 표시해 준다.)

* 최종 목표 : 논문 작성 -> 휴대폰 카메라로 일부 피부 사진을 촬영하면 아토피 위치를 정확히 알려주는 어플리케이션 개발
* 문제점 : 데이터 부족 (개인 동의, 인력 등)
* 보안문제로 인해 데이터는 제공할 수 없음
* 슈퍼컴퓨터를 이용하여 multi-gpu 사용

1. data preprocessing : atopy-1, skin-0, background-2
2. Model : vgg, resnet (framework : keras)
3. object detection : yolo algorism
    - 대부분 CNN은 Channel을 계속 줄여나가다가 flatten을 수행한다. 하지만 flatten을 수행하지 않고 1x1x1 tensor만을 남기고, training을 진행
    - tensor가 아토피라는 결론이라면 원래의 이미지에서 최종 tensor의 위치를 수학적으로 계산하여 그 크기만큼 표시 해 준다.


