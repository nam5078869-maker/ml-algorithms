#ifndef KMEANS_H
#define KMEANS_H

#include "matrix.h"

/*
 * k-means 군집화 (비지도학습)
 *
 * 정답 없이 데이터를 k개 묶음으로 나눈다.
 *   1) 중심 k개를 정한다
 *   2) 각 점을 가장 가까운 중심에 배정한다
 *   3) 각 중심을 배정된 점들의 평균 위치로 옮긴다
 *   4) 배정이 더 이상 바뀌지 않을 때까지 2~3을 반복한다
 */

typedef enum {
    INIT_RANDOM,       /* 데이터 중 k개를 무작위로 골라 중심으로 */
    INIT_PLUSPLUS      /* k-means++: 서로 멀리 떨어지도록 골라서 */
} KMeansInit;

typedef struct {
    int     k;
    int     d;
    Matrix *centroids;   /* k x d 중심 좌표 */
    int    *assign;      /* 각 점이 속한 군집 번호 (길이 n) */
    int     n;
    int     iterations;  /* 수렴까지 반복한 횟수 */
    double  inertia;     /* 각 점과 자기 중심 사이 거리 제곱의 합 (작을수록 촘촘) */
} KMeans;

/* 모델을 만든다. n은 학습할 점의 개수 */
KMeans *kmeans_create(int k, int d, int n);
void    kmeans_free(KMeans *model);

/* 중심만 초기화한다 (반복 과정을 한 단계씩 보고 싶을 때) */
void kmeans_init_centroids(KMeans *model, const Matrix *X, KMeansInit init);

/* 배정 단계 한 번. 배정이 바뀐 점의 개수를 돌려준다 */
int kmeans_assign_step(KMeans *model, const Matrix *X);

/* 중심 이동 단계 한 번 */
void kmeans_update_step(KMeans *model, const Matrix *X);

/* 초기화부터 수렴까지 한 번에 실행한다. 반복 횟수를 돌려준다 */
int kmeans_fit(KMeans *model, const Matrix *X, KMeansInit init, int max_iter);

/* 점 하나가 가장 가까운 중심 번호 */
int kmeans_nearest(const KMeans *model, const double *x);

#endif
