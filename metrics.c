#include <math.h>
#include "metrics.h"

double metric_mse(const Matrix *pred, const Matrix *y) {
    double sum = 0.0;
    for (int i = 0; i < y->rows; i++) {
        double diff = MAT_AT(pred, i, 0) - MAT_AT(y, i, 0);
        sum += diff * diff;
    }
    return sum / y->rows;
}

double metric_rmse(const Matrix *pred, const Matrix *y) {
    return sqrt(metric_mse(pred, y));
}

/*
 * R² = 1 - (모델 오차의 제곱합) / (평균과의 차이 제곱합)
 * 분모는 "모든 값을 평균으로 예측했을 때"의 오차다.
 * 모델이 평균 찍기보다 얼마나 나은지를 0~1로 보여준다.
 */
double metric_r2(const Matrix *pred, const Matrix *y) {
    double mean = 0.0;
    for (int i = 0; i < y->rows; i++) {
        mean += MAT_AT(y, i, 0);
    }
    mean /= y->rows;

    double ss_res = 0.0, ss_tot = 0.0;
    for (int i = 0; i < y->rows; i++) {
        double e = MAT_AT(y, i, 0) - MAT_AT(pred, i, 0);
        double t = MAT_AT(y, i, 0) - mean;
        ss_res += e * e;
        ss_tot += t * t;
    }
    if (ss_tot < 1e-12) {
        return 0.0;
    }
    return 1.0 - ss_res / ss_tot;
}

double metric_accuracy(const int *pred, const int *label, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (pred[i] == label[i]) {
            correct++;
        }
    }
    return 100.0 * correct / n;
}
