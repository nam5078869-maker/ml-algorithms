#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "kmeans.h"

static double squared_distance(const double *a, const double *b, int d) {
    double sum = 0.0;
    for (int j = 0; j < d; j++) {
        double diff = a[j] - b[j];
        sum += diff * diff;
    }
    return sum;
}

/* 0 이상 1 미만의 균등분포 난수 */
static double random_unit(void) {
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

KMeans *kmeans_create(int k, int d, int n) {
    KMeans *model = calloc(1, sizeof(KMeans));
    if (model == NULL) {
        return NULL;
    }
    model->k = k;
    model->d = d;
    model->n = n;
    model->centroids = mat_create(k, d);
    model->assign = malloc(sizeof(int) * (size_t)n);
    if (model->centroids == NULL || model->assign == NULL) {
        kmeans_free(model);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        model->assign[i] = -1;        /* 아직 배정 안 됨 */
    }
    return model;
}

void kmeans_free(KMeans *model) {
    if (model == NULL) {
        return;
    }
    mat_free(model->centroids);
    free(model->assign);
    free(model);
}

int kmeans_nearest(const KMeans *model, const double *x) {
    int best = 0;
    double best_dist = DBL_MAX;
    for (int c = 0; c < model->k; c++) {
        double dist = squared_distance(x, &MAT_AT(model->centroids, c, 0),
                                       model->d);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    return best;
}

/* ================= 초기화 ================= */

static void copy_point(KMeans *model, int c, const Matrix *X, int i) {
    memcpy(&MAT_AT(model->centroids, c, 0), &MAT_AT(X, i, 0),
           sizeof(double) * (size_t)model->d);
}

/*
 * k-means++ 초기화
 *
 *   1) 첫 중심은 무작위로 고른다
 *   2) 다음 중심은 "이미 고른 중심들과의 거리²"에 비례하는 확률로 고른다
 *      → 멀리 떨어진 점일수록 뽑힐 확률이 높다
 *   3) k개가 될 때까지 반복
 *
 * 중심들이 한곳에 몰려 시작하는 일을 막아 결과가 훨씬 안정적이다.
 */
static void init_plusplus(KMeans *model, const Matrix *X) {
    int n = X->rows;
    double *min_dist = malloc(sizeof(double) * (size_t)n);
    if (min_dist == NULL) {
        return;
    }

    copy_point(model, 0, X, rand() % n);
    for (int i = 0; i < n; i++) {
        min_dist[i] = squared_distance(&MAT_AT(X, i, 0),
                                       &MAT_AT(model->centroids, 0, 0),
                                       model->d);
    }

    for (int c = 1; c < model->k; c++) {
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            total += min_dist[i];
        }

        /* 룰렛 휠: 0~total 사이 값을 뽑아, 누적합이 넘는 점을 고른다 */
        double target = random_unit() * total;
        int chosen = n - 1;
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += min_dist[i];
            if (acc > target) {
                chosen = i;
                break;
            }
        }
        copy_point(model, c, X, chosen);

        /* 새 중심이 생겼으니 각 점의 "가장 가까운 중심까지 거리"를 갱신 */
        for (int i = 0; i < n; i++) {
            double dist = squared_distance(&MAT_AT(X, i, 0),
                                           &MAT_AT(model->centroids, c, 0),
                                           model->d);
            if (dist < min_dist[i]) {
                min_dist[i] = dist;
            }
        }
    }
    free(min_dist);
}

static void init_random(KMeans *model, const Matrix *X) {
    int n = X->rows;
    /* 서로 다른 점 k개를 고른다 (피셔-예이츠를 앞 k개까지만) */
    int *order = malloc(sizeof(int) * (size_t)n);
    if (order == NULL) {
        return;
    }
    for (int i = 0; i < n; i++) {
        order[i] = i;
    }
    for (int c = 0; c < model->k; c++) {
        int j = c + rand() % (n - c);
        int tmp = order[c];
        order[c] = order[j];
        order[j] = tmp;
        copy_point(model, c, X, order[c]);
    }
    free(order);
}

void kmeans_init_centroids(KMeans *model, const Matrix *X, KMeansInit init) {
    if (init == INIT_PLUSPLUS) {
        init_plusplus(model, X);
    } else {
        init_random(model, X);
    }
    for (int i = 0; i < model->n; i++) {
        model->assign[i] = -1;
    }
    model->iterations = 0;
}

/* ================= 반복 ================= */

int kmeans_assign_step(KMeans *model, const Matrix *X) {
    int changed = 0;
    double inertia = 0.0;

    for (int i = 0; i < X->rows; i++) {
        const double *x = &MAT_AT(X, i, 0);
        int best = kmeans_nearest(model, x);
        if (best != model->assign[i]) {
            model->assign[i] = best;
            changed++;
        }
        inertia += squared_distance(x, &MAT_AT(model->centroids, best, 0),
                                    model->d);
    }
    model->inertia = inertia;
    return changed;
}

void kmeans_update_step(KMeans *model, const Matrix *X) {
    int k = model->k, d = model->d;
    int *count = calloc((size_t)k, sizeof(int));
    if (count == NULL) {
        return;
    }

    /* 새 중심 = 배정된 점들의 평균. 기존 값을 버리고 합부터 다시 구한다 */
    Matrix *old = mat_create(k, d);
    mat_copy(old, model->centroids);
    mat_fill(model->centroids, 0.0);

    for (int i = 0; i < X->rows; i++) {
        int c = model->assign[i];
        count[c]++;
        for (int j = 0; j < d; j++) {
            MAT_AT(model->centroids, c, j) += MAT_AT(X, i, j);
        }
    }

    for (int c = 0; c < k; c++) {
        if (count[c] == 0) {
            /* 아무 점도 배정되지 않은 중심은 원래 자리에 둔다 */
            memcpy(&MAT_AT(model->centroids, c, 0), &MAT_AT(old, c, 0),
                   sizeof(double) * (size_t)d);
            continue;
        }
        for (int j = 0; j < d; j++) {
            MAT_AT(model->centroids, c, j) /= count[c];
        }
    }

    mat_free(old);
    free(count);
}

int kmeans_fit(KMeans *model, const Matrix *X, KMeansInit init, int max_iter) {
    kmeans_init_centroids(model, X, init);

    int iter;
    for (iter = 1; iter <= max_iter; iter++) {
        int changed = kmeans_assign_step(model, X);
        if (changed == 0) {
            break;                    /* 배정이 그대로면 수렴 */
        }
        kmeans_update_step(model, X);
    }
    model->iterations = iter > max_iter ? max_iter : iter;
    return model->iterations;
}
