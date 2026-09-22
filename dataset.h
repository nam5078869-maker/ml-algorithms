#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"

/*
 * 머신러닝 실험용 데이터 묶음.
 *
 *   X: n개 샘플 x d개 특징 (행 하나가 샘플 하나)
 *   y: 회귀용 목표값 (n x 1), 분류 데이터면 NULL
 *   label: 분류용 정답 (n개), 회귀 데이터면 NULL
 */
typedef struct {
    Matrix *X;
    Matrix *y;
    int    *label;
    int     n;          /* 샘플 수 */
    int     d;          /* 특징 수 */
    int     n_classes;  /* 분류일 때 클래스 수, 회귀면 0 */
} Data;

/* ---------- 생성과 해제 ---------- */

/* 빈 데이터를 만든다. with_y / with_label이 1이면 해당 배열도 할당한다 */
Data *data_create(int n, int d, int with_y, int with_label);
void  data_free(Data *data);

/* ---------- 가상 데이터 생성 ---------- */

/*
 * 선형 회귀용 데이터.
 *   y = X·w_true + b_true + 잡음
 * 실제 사용한 w와 b는 true_w(길이 d), true_b에 담아 돌려준다.
 * 알고리즘이 정답을 얼마나 잘 찾아내는지 비교할 수 있다.
 */
Data *data_make_regression(int n, int d, double noise,
                           double *true_w, double *true_b);

/*
 * 군집/분류용 데이터.
 * k개의 중심 주변에 정규분포로 점들을 흩뿌린다 (blob).
 * spread가 클수록 군집이 서로 섞인다.
 */
Data *data_make_blobs(int n, int d, int k, double spread);

/* ---------- 파일 ---------- */

/* CSV로 저장한다. 마지막 열은 y(회귀) 또는 label(분류) */
int data_save_csv(const Data *data, const char *path);

/* CSV를 읽는다. is_label이 1이면 마지막 열을 정수 라벨로 읽는다 */
Data *data_load_csv(const char *path, int is_label);

/* ---------- 전처리 ---------- */

/*
 * 데이터를 섞은 뒤 학습용과 시험용으로 나눈다.
 * train_ratio 0.8이면 80%가 학습용.
 */
int data_split(const Data *src, double train_ratio,
               Data **train, Data **test);

/*
 * 특징별로 평균 0, 표준편차 1이 되도록 바꾼다 (표준화).
 * 시험 데이터는 반드시 학습 데이터의 평균·표준편차로 변환해야 한다.
 */
void data_standardize(Data *train, Data *test);

/* ---------- 확인 ---------- */

void data_summary(const Data *data, const char *name);

/* 특징 두 개를 골라 터미널에 산점도를 그린다.
 * 분류 데이터면 클래스별로 다른 문자로 표시한다 */
void data_scatter(const Data *data, int feature_x, int feature_y);

/* 표준정규분포 난수 (평균 0, 표준편차 1) */
double random_normal(void);

#endif
