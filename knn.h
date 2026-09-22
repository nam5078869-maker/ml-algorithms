#ifndef KNN_H
#define KNN_H

#include "matrix.h"

/*
 * k-최근접 이웃 (KNN) 분류
 *
 * "학습"이 없다. 학습 데이터를 그대로 기억해 두었다가,
 * 새 점이 들어오면 가장 가까운 k개를 찾아 다수결로 정한다.
 */
typedef struct {
    const Matrix *X;       /* 학습 데이터 (복사하지 않고 가리키기만 한다) */
    const int    *label;   /* 학습 데이터 정답 */
    int           n_classes;
    int           k;
} KNN;

/* 학습 데이터와 k를 정해 모델을 준비한다 (계산은 하지 않는다) */
void knn_init(KNN *model, const Matrix *X, const int *label,
              int n_classes, int k);

/* 점 하나(길이 d 배열)의 클래스를 예측한다 */
int knn_predict_one(const KNN *model, const double *x);

/* 여러 점을 한꺼번에 예측한다. 결과는 pred(길이 X->rows)에 담긴다.
 * 성공 0, 메모리 부족 -1 */
int knn_predict(const KNN *model, const Matrix *X, int *pred);

/*
 * k겹 교차검증 정확도(%)를 계산한다.
 * 데이터를 folds개로 나눠, 한 조각씩 돌아가며 검증용으로 쓴다.
 * 시험 데이터를 건드리지 않고 k를 고르기 위해 사용한다.
 */
double knn_cross_validate(const Matrix *X, const int *label, int n_classes,
                          int k, int folds);

#endif
