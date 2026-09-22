#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "linreg.h"

LinReg *linreg_create(int d) {
    LinReg *model = calloc(1, sizeof(LinReg));
    if (model == NULL) {
        return NULL;
    }
    model->w = calloc((size_t)d, sizeof(double));
    if (model->w == NULL) {
        free(model);
        return NULL;
    }
    model->d = d;
    return model;
}

void linreg_free(LinReg *model) {
    if (model == NULL) {
        return;
    }
    free(model->w);
    free(model);
}

void linreg_predict(const LinReg *model, const Matrix *X, Matrix *pred) {
    for (int i = 0; i < X->rows; i++) {
        double sum = model->b;
        for (int j = 0; j < model->d; j++) {
            sum += MAT_AT(X, i, j) * model->w[j];
        }
        MAT_AT(pred, i, 0) = sum;
    }
}

/* ================= 경사하강법 ================= */

/*
 * 손실: L = (1/n) Σ (ŷᵢ - yᵢ)²
 * 미분:
 *   ∂L/∂wⱼ = (2/n) Σ (ŷᵢ - yᵢ) · xᵢⱼ
 *   ∂L/∂b  = (2/n) Σ (ŷᵢ - yᵢ)
 */
double linreg_fit_gd(LinReg *model, const Matrix *X, const Matrix *y,
                     double learning_rate, int epochs, double *loss_history) {
    int n = X->rows;
    int d = model->d;

    double *grad_w = malloc(sizeof(double) * (size_t)d);
    Matrix *pred = mat_create(n, 1);
    if (grad_w == NULL || pred == NULL) {
        free(grad_w);
        mat_free(pred);
        return -1.0;
    }

    /* 0에서 시작한다 */
    memset(model->w, 0, sizeof(double) * (size_t)d);
    model->b = 0.0;

    double loss = 0.0;
    double first_loss = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        linreg_predict(model, X, pred);

        memset(grad_w, 0, sizeof(double) * (size_t)d);
        double grad_b = 0.0;
        loss = 0.0;

        for (int i = 0; i < n; i++) {
            double err = MAT_AT(pred, i, 0) - MAT_AT(y, i, 0);
            loss += err * err;
            for (int j = 0; j < d; j++) {
                grad_w[j] += err * MAT_AT(X, i, j);
            }
            grad_b += err;
        }
        loss /= n;

        if (loss_history != NULL) {
            loss_history[epoch] = loss;
        }
        if (epoch == 0) {
            first_loss = loss;
        }
        /* 학습률이 너무 크면 손실이 줄지 않고 폭발한다.
         * 처음보다 100만 배 이상 커지면 발산으로 보고 멈춘다 */
        if (!isfinite(loss) || loss > first_loss * 1e6) {
            free(grad_w);
            mat_free(pred);
            return -1.0;
        }

        for (int j = 0; j < d; j++) {
            model->w[j] -= learning_rate * (2.0 / n) * grad_w[j];
        }
        model->b -= learning_rate * (2.0 / n) * grad_b;
    }

    free(grad_w);
    mat_free(pred);
    return loss;
}

/* ================= 정규방정식 ================= */

/*
 * 가우스 소거법으로 연립방정식 M·x = v 를 푼다 (M: k x k).
 * M과 v는 계산 중에 바뀐다. 결과는 v에 담긴다.
 *
 * 부분 피벗팅: 각 열에서 절댓값이 가장 큰 행을 위로 올린다.
 * 아주 작은 수로 나누면 오차가 커지기 때문이다.
 */
static int gaussian_solve(Matrix *M, double *v) {
    int k = M->rows;

    for (int col = 0; col < k; col++) {
        /* 1) 피벗 찾기 */
        int pivot = col;
        for (int r = col + 1; r < k; r++) {
            if (fabs(MAT_AT(M, r, col)) > fabs(MAT_AT(M, pivot, col))) {
                pivot = r;
            }
        }
        if (fabs(MAT_AT(M, pivot, col)) < 1e-12) {
            return -1;                  /* 풀 수 없음 (특이 행렬) */
        }

        /* 2) 피벗 행을 위로 */
        if (pivot != col) {
            for (int c = 0; c < k; c++) {
                double tmp = MAT_AT(M, col, c);
                MAT_AT(M, col, c) = MAT_AT(M, pivot, c);
                MAT_AT(M, pivot, c) = tmp;
            }
            double tmp = v[col];
            v[col] = v[pivot];
            v[pivot] = tmp;
        }

        /* 3) 아래 행들의 이 열을 0으로 만든다 */
        for (int r = col + 1; r < k; r++) {
            double factor = MAT_AT(M, r, col) / MAT_AT(M, col, col);
            for (int c = col; c < k; c++) {
                MAT_AT(M, r, c) -= factor * MAT_AT(M, col, c);
            }
            v[r] -= factor * v[col];
        }
    }

    /* 4) 뒤에서부터 대입 (후진 대입) */
    for (int r = k - 1; r >= 0; r--) {
        double sum = v[r];
        for (int c = r + 1; c < k; c++) {
            sum -= MAT_AT(M, r, c) * v[c];
        }
        v[r] = sum / MAT_AT(M, r, r);
    }
    return 0;
}

int linreg_fit_normal(LinReg *model, const Matrix *X, const Matrix *y) {
    int n = X->rows;
    int d = model->d;
    int k = d + 1;                      /* 편향까지 포함한 미지수 수 */

    /*
     * A = [X | 1] 이라 하면
     *   AᵀA 의 (p, q) 원소 = Σᵢ A[i][p] · A[i][q]
     *   Aᵀy 의 p 원소     = Σᵢ A[i][p] · y[i]
     * A를 실제로 만들지 않고 바로 계산한다 (마지막 열은 항상 1).
     */
    Matrix *AtA = mat_create(k, k);
    double *Aty = calloc((size_t)k, sizeof(double));
    if (AtA == NULL || Aty == NULL) {
        mat_free(AtA);
        free(Aty);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        for (int p = 0; p < k; p++) {
            double a_p = p < d ? MAT_AT(X, i, p) : 1.0;
            for (int q = 0; q < k; q++) {
                double a_q = q < d ? MAT_AT(X, i, q) : 1.0;
                MAT_AT(AtA, p, q) += a_p * a_q;
            }
            Aty[p] += a_p * MAT_AT(y, i, 0);
        }
    }

    int result = gaussian_solve(AtA, Aty);
    if (result == 0) {
        memcpy(model->w, Aty, sizeof(double) * (size_t)d);
        model->b = Aty[d];
    }

    mat_free(AtA);
    free(Aty);
    return result;
}
