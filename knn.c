#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "knn.h"
#include "metrics.h"

#define MAX_K 256

void knn_init(KNN *model, const Matrix *X, const int *label,
              int n_classes, int k) {
    model->X = X;
    model->label = label;
    model->n_classes = n_classes;
    model->k = k < 1 ? 1 : (k > MAX_K ? MAX_K : k);
}

/*
 * 두 점 사이 거리의 "제곱"을 계산한다.
 * 가까운 순서만 알면 되므로 제곱근(sqrt)은 생략해도 순서가 같다.
 * 모든 점마다 sqrt를 계산하지 않아도 되니 빨라진다.
 */
static double squared_distance(const double *a, const double *b, int d) {
    double sum = 0.0;
    for (int j = 0; j < d; j++) {
        double diff = a[j] - b[j];
        sum += diff * diff;
    }
    return sum;
}

int knn_predict_one(const KNN *model, const double *x) {
    int k = model->k;
    int n = model->X->rows;
    int d = model->X->cols;
    if (k > n) {
        k = n;
    }

    /*
     * 가장 가까운 k개를 거리 순으로 유지하는 작은 배열.
     * 전체를 정렬(n log n)하지 않고, 새 점이 k번째보다 가까울 때만
     * 제자리에 끼워 넣는다 (삽입 정렬). k가 작으면 훨씬 빠르다.
     */
    double best_dist[MAX_K];
    int    best_label[MAX_K] = {0};
    int    count = 0;

    for (int i = 0; i < n; i++) {
        double dist = squared_distance(x, &MAT_AT(model->X, i, 0), d);

        if (count < k) {
            count++;
        } else if (dist >= best_dist[k - 1]) {
            continue;                   /* k번째보다 멀면 볼 필요 없음 */
        }

        /* 뒤에서부터 한 칸씩 밀며 들어갈 자리를 찾는다 */
        int pos = count - 1;
        while (pos > 0 && best_dist[pos - 1] > dist) {
            best_dist[pos]  = best_dist[pos - 1];
            best_label[pos] = best_label[pos - 1];
            pos--;
        }
        best_dist[pos]  = dist;
        best_label[pos] = model->label[i];
    }

    /* 다수결. 표가 같으면 더 가까운 이웃이 속한 클래스를 고른다 */
    int votes[256] = {0};
    int best = best_label[0];
    for (int i = 0; i < count; i++) {
        int c = best_label[i];
        votes[c]++;
        if (votes[c] > votes[best]) {
            best = c;
        }
    }
    return best;
}

int knn_predict(const KNN *model, const Matrix *X, int *pred) {
    for (int i = 0; i < X->rows; i++) {
        pred[i] = knn_predict_one(model, &MAT_AT(X, i, 0));
    }
    return 0;
}

double knn_cross_validate(const Matrix *X, const int *label, int n_classes,
                          int k, int folds) {
    int n = X->rows;
    int d = X->cols;
    int fold_size = n / folds;

    Matrix *train_X = mat_create(n - fold_size, d);
    int *train_label = malloc(sizeof(int) * (size_t)n);
    int *pred = malloc(sizeof(int) * (size_t)fold_size);
    if (train_X == NULL || train_label == NULL || pred == NULL) {
        mat_free(train_X);
        free(train_label);
        free(pred);
        return -1.0;
    }

    double total = 0.0;
    for (int f = 0; f < folds; f++) {
        int start = f * fold_size;          /* 이번 검증 조각의 시작 */

        /* 검증 조각을 뺀 나머지를 학습 데이터로 모은다 */
        int row = 0;
        for (int i = 0; i < n && row < n - fold_size; i++) {
            if (i >= start && i < start + fold_size) {
                continue;
            }
            memcpy(&MAT_AT(train_X, row, 0), &MAT_AT(X, i, 0),
                   sizeof(double) * (size_t)d);
            train_label[row] = label[i];
            row++;
        }

        KNN model;
        knn_init(&model, train_X, train_label, n_classes, k);
        for (int i = 0; i < fold_size; i++) {
            pred[i] = knn_predict_one(&model, &MAT_AT(X, start + i, 0));
        }
        total += metric_accuracy(pred, &label[start], fold_size);
    }

    mat_free(train_X);
    free(train_label);
    free(pred);
    return total / folds;
}
