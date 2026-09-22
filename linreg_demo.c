/*
 * 2단계: 선형 회귀
 *
 * 정답을 알고 만든 데이터로 두 방법을 비교한다.
 *   1) 경사하강법   2) 정규방정식
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dataset.h"
#include "linreg.h"
#include "metrics.h"

#define SEED 42

static double seconds_since(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

/* 학습/시험 데이터에 대한 지표를 한 줄로 출력 */
static void report(const char *name, const LinReg *m,
                   const Data *train, const Data *test) {
    Matrix *p_train = mat_create(train->n, 1);
    Matrix *p_test  = mat_create(test->n, 1);
    linreg_predict(m, train->X, p_train);
    linreg_predict(m, test->X, p_test);

    printf("%-12s 학습 MSE %.4f | 시험 MSE %.4f, RMSE %.4f, R² %.4f\n",
           name, metric_mse(p_train, train->y),
           metric_mse(p_test, test->y), metric_rmse(p_test, test->y),
           metric_r2(p_test, test->y));

    mat_free(p_train);
    mat_free(p_test);
}

/* ---------- 실험 1: 1차원 데이터와 회귀선 그리기 ---------- */

#define PW 60
#define PH 18

static void plot_line(const Data *data, const LinReg *m) {
    double min_x = 1e18, max_x = -1e18, min_y = 1e18, max_y = -1e18;
    for (int i = 0; i < data->n; i++) {
        double x = MAT_AT(data->X, i, 0), y = MAT_AT(data->y, i, 0);
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }

    char canvas[PH][PW + 1];
    memset(canvas, ' ', sizeof(canvas));
    for (int r = 0; r < PH; r++) canvas[r][PW] = '\0';

    /* 회귀선 먼저 */
    for (int c = 0; c < PW; c++) {
        double x = min_x + (max_x - min_x) * c / (PW - 1);
        double y = m->w[0] * x + m->b;
        int r = PH - 1 - (int)((y - min_y) / (max_y - min_y) * (PH - 1) + 0.5);
        if (r >= 0 && r < PH) canvas[r][c] = '-';
    }
    /* 데이터 점 */
    for (int i = 0; i < data->n; i++) {
        int c = (int)((MAT_AT(data->X, i, 0) - min_x) / (max_x - min_x) * (PW - 1));
        int r = PH - 1 - (int)((MAT_AT(data->y, i, 0) - min_y) / (max_y - min_y) * (PH - 1));
        canvas[r][c] = canvas[r][c] == '-' ? '#' : 'o';
    }

    printf("가로 x (%.1f ~ %.1f), 세로 y (%.1f ~ %.1f)  o=데이터  -=회귀선\n",
           min_x, max_x, min_y, max_y);
    for (int r = 0; r < PH; r++) printf("|%s|\n", canvas[r]);
}

static void experiment_1d(void) {
    printf("\n==================== 실험 1: 특징 1개 ====================\n");
    double tw, tb;
    Data *data = data_make_regression(80, 1, 1.0, &tw, &tb);
    LinReg *m = linreg_create(1);
    linreg_fit_normal(m, data->X, data->y);

    printf("실제 식:   y = %.3f x + %.3f\n", tw, tb);
    printf("찾은 식:   y = %.3f x + %.3f\n\n", m->w[0], m->b);
    plot_line(data, m);

    linreg_free(m);
    data_free(data);
}

/* ---------- 실험 2: 특징 3개, 두 방법 비교 ---------- */

static void experiment_compare(void) {
    printf("\n==================== 실험 2: 두 방법 비교 ====================\n");
    const int d = 3;
    double tw[3], tb;
    Data *all = data_make_regression(1000, d, 0.5, tw, &tb);
    Data *train, *test;
    data_split(all, 0.8, &train, &test);

    LinReg *gd = linreg_create(d);
    LinReg *ne = linreg_create(d);

    const int epochs = 200;
    double history[200];
    linreg_fit_gd(gd, train->X, train->y, 0.05, epochs, history);
    linreg_fit_normal(ne, train->X, train->y);

    printf("            w1        w2        w3        b\n");
    printf("실제값   %8.4f  %8.4f  %8.4f  %8.4f\n", tw[0], tw[1], tw[2], tb);
    printf("경사하강 %8.4f  %8.4f  %8.4f  %8.4f\n", gd->w[0], gd->w[1], gd->w[2], gd->b);
    printf("정규방정 %8.4f  %8.4f  %8.4f  %8.4f\n\n", ne->w[0], ne->w[1], ne->w[2], ne->b);

    report("경사하강법", gd, train, test);
    report("정규방정식", ne, train, test);
    printf("(잡음의 분산이 0.25이므로 MSE는 0.25 근처가 최선이다)\n");

    printf("\n[경사하강법 손실 곡선]\n");
    int marks[] = {0, 1, 2, 5, 10, 20, 50, 100, 199};
    for (int i = 0; i < 9; i++) {
        int e = marks[i];
        printf("  에폭 %3d  MSE %8.4f  ", e + 1, history[e]);
        int bars = (int)(history[e] / history[0] * 50);
        for (int b = 0; b < bars; b++) putchar('#');
        putchar('\n');
    }

    linreg_free(gd);
    linreg_free(ne);
    data_free(all);
    data_free(train);
    data_free(test);
}

/* ---------- 실험 3: 학습률 ---------- */

static void experiment_learning_rate(void) {
    printf("\n==================== 실험 3: 학습률 ====================\n");
    Data *data = data_make_regression(500, 3, 0.5, NULL, NULL);
    LinReg *m = linreg_create(3);

    double rates[] = {0.001, 0.01, 0.1, 0.5, 1.1};
    printf("학습률   50에폭 후 MSE\n");
    for (int i = 0; i < 5; i++) {
        double loss = linreg_fit_gd(m, data->X, data->y, rates[i], 50, NULL);
        if (loss < 0) {
            printf("%6.3f   발산 (값이 폭발함)\n", rates[i]);
        } else {
            printf("%6.3f   %10.4f\n", rates[i], loss);
        }
    }

    linreg_free(m);
    data_free(data);
}

/* ---------- 실험 4: 특징이 많을 때 속도 ---------- */

static void experiment_speed(void) {
    printf("\n==================== 실험 4: 특징 수에 따른 속도 ====================\n");
    printf("특징 수   정규방정식   경사하강법(100에폭)\n");

    int dims[] = {10, 100, 300};
    for (int i = 0; i < 3; i++) {
        int d = dims[i];
        Data *data = data_make_regression(2000, d, 0.5, NULL, NULL);
        LinReg *m = linreg_create(d);

        clock_t t = clock();
        linreg_fit_normal(m, data->X, data->y);
        double t_ne = seconds_since(t);

        t = clock();
        linreg_fit_gd(m, data->X, data->y, 0.1, 100, NULL);
        double t_gd = seconds_since(t);

        printf("%6d   %9.3f초   %9.3f초\n", d, t_ne, t_gd);
        linreg_free(m);
        data_free(data);
    }
    printf("정규방정식은 특징 수 d에 대해 약 d³, 경사하강법은 d에 비례해 늘어난다.\n");
}

int main(void) {
    srand(SEED);
    experiment_1d();
    experiment_compare();
    experiment_learning_rate();
    experiment_speed();
    return 0;
}
