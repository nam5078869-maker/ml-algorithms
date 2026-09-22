/*
 * 1단계: 실험용 데이터를 만들고 확인한다.
 *
 * 알고리즘을 만들기 전에 "정답을 아는 데이터"를 준비하는 것이 중요하다.
 * 예를 들어 y = 2.5*x1 - 1.3*x2 + 0.7 로 만든 데이터라면,
 * 선형 회귀가 그 값을 찾아내는지 확인할 수 있다.
 */
#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"

#define SEED 42

int main(void) {
    srand(SEED);

    /* ---- 회귀용 데이터 ---- */
    double true_w[2], true_b;
    Data *reg = data_make_regression(300, 2, 0.5, true_w, &true_b);
    if (reg == NULL) {
        return 1;
    }

    printf("=== 회귀 데이터 ===\n");
    printf("실제 식: y = %.3f*x1 + %.3f*x2 + %.3f + 잡음(표준편차 0.5)\n",
           true_w[0], true_w[1], true_b);
    data_summary(reg, "회귀 데이터");

    /* ---- 군집/분류용 데이터 ---- */
    Data *blobs = data_make_blobs(300, 2, 3, 1.5);
    if (blobs == NULL) {
        data_free(reg);
        return 1;
    }

    printf("\n=== 군집 데이터 ===\n");
    data_summary(blobs, "군집 데이터");
    data_scatter(blobs, 0, 1);

    /* ---- CSV 저장하고 다시 읽어서 확인 ---- */
    printf("\n=== CSV 저장/불러오기 ===\n");
    if (data_save_csv(reg, "regression.csv") != 0
        || data_save_csv(blobs, "blobs.csv") != 0) {
        printf("저장 실패\n");
    } else {
        printf("regression.csv, blobs.csv 저장 완료\n");

        Data *loaded = data_load_csv("blobs.csv", 1);
        if (loaded != NULL) {
            printf("다시 읽은 결과: 샘플 %d개, 특징 %d개, 클래스 %d개\n",
                   loaded->n, loaded->d, loaded->n_classes);
            printf("첫 샘플 값 비교: 원본 (%.6f, %.6f) / 읽은 값 (%.6f, %.6f)\n",
                   MAT_AT(blobs->X, 0, 0), MAT_AT(blobs->X, 0, 1),
                   MAT_AT(loaded->X, 0, 0), MAT_AT(loaded->X, 0, 1));
            data_free(loaded);
        }
    }

    /* ---- 학습/시험 분할과 표준화 ---- */
    printf("\n=== 학습/시험 분할 (8:2) ===\n");
    Data *train = NULL, *test = NULL;
    if (data_split(blobs, 0.8, &train, &test) == 0) {
        data_summary(train, "학습용");
        data_summary(test, "시험용");

        printf("\n=== 표준화 후 (평균 0, 표준편차 1) ===\n");
        data_standardize(train, test);
        data_summary(train, "학습용(표준화)");
        data_summary(test, "시험용(표준화: 학습 데이터 기준)");

        data_free(train);
        data_free(test);
    }

    data_free(reg);
    data_free(blobs);
    return 0;
}
