#ifndef LINREG_H
#define LINREG_H

#include "matrix.h"

/*
 * 선형 회귀: ŷ = X·w + b
 *
 * 같은 문제를 두 가지 방법으로 푼다.
 *  1) 경사하강법: 손실을 조금씩 줄여가며 반복해서 찾는다
 *  2) 정규방정식: 미분 = 0 인 지점을 식으로 바로 계산한다
 */
typedef struct {
    int     d;       /* 특징 수 */
    double *w;       /* 가중치 d개 */
    double  b;       /* 편향 */
} LinReg;

LinReg *linreg_create(int d);
void    linreg_free(LinReg *model);

/* ŷ = X·w + b 를 계산해 pred(n x 1)에 저장한다 */
void linreg_predict(const LinReg *model, const Matrix *X, Matrix *pred);

/*
 * 경사하강법으로 학습한다.
 * loss_history가 NULL이 아니면 에폭마다 학습 MSE를 기록한다 (길이 epochs).
 * 반환값: 마지막 MSE. 발산하면(값이 무한대가 되면) -1
 */
double linreg_fit_gd(LinReg *model, const Matrix *X, const Matrix *y,
                     double learning_rate, int epochs, double *loss_history);

/*
 * 정규방정식으로 한 번에 푼다: (AᵀA)θ = Aᵀy
 * (A는 X 오른쪽에 1로 된 열을 붙인 행렬, θ = [w; b])
 * 성공 0, 행렬이 특이해서 풀 수 없으면 -1
 */
int linreg_fit_normal(LinReg *model, const Matrix *X, const Matrix *y);

#endif
