#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"

/* ---------- 회귀 평가 지표 (pred, y: n x 1) ---------- */

/* 평균 제곱 오차: 평균[(예측 - 정답)²] */
double metric_mse(const Matrix *pred, const Matrix *y);

/* MSE의 제곱근. 정답과 같은 단위라 해석하기 쉽다 */
double metric_rmse(const Matrix *pred, const Matrix *y);

/* 결정계수 R²: 1이면 완벽, 0이면 "평균으로 찍기"와 같은 수준 */
double metric_r2(const Matrix *pred, const Matrix *y);

/* ---------- 분류 평가 지표 ---------- */

/* 정확도(%): 맞힌 개수 / 전체 */
double metric_accuracy(const int *pred, const int *label, int n);

#endif
