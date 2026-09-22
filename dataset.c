#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataset.h"

#define LINE_SIZE 4096

/* -std=c11에서는 math.h의 M_PI가 없을 수 있어 직접 정의한다 */
#define PI 3.14159265358979323846

/*
 * 박스-뮐러 변환: 균등분포 난수 두 개로 정규분포 난수를 만든다.
 *   z = sqrt(-2 ln u1) * cos(2π u2)
 * 한 번 계산하면 값이 두 개 나오므로 하나는 저장해 두고 다음에 쓴다.
 */
double random_normal(void) {
    static int have_spare = 0;
    static double spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    double u1, u2;
    do {
        u1 = (double)rand() / RAND_MAX;
    } while (u1 <= 1e-12);            /* log(0) 방지 */
    u2 = (double)rand() / RAND_MAX;

    double radius = sqrt(-2.0 * log(u1));
    double theta  = 2.0 * PI * u2;

    spare = radius * sin(theta);
    have_spare = 1;
    return radius * cos(theta);
}

/* ================= 생성과 해제 ================= */

Data *data_create(int n, int d, int with_y, int with_label) {
    Data *data = calloc(1, sizeof(Data));
    if (data == NULL) {
        return NULL;
    }

    data->n = n;
    data->d = d;
    data->X = mat_create(n, d);
    if (data->X == NULL) {
        data_free(data);
        return NULL;
    }
    if (with_y) {
        data->y = mat_create(n, 1);
        if (data->y == NULL) {
            data_free(data);
            return NULL;
        }
    }
    if (with_label) {
        data->label = calloc((size_t)n, sizeof(int));
        if (data->label == NULL) {
            data_free(data);
            return NULL;
        }
    }
    return data;
}

void data_free(Data *data) {
    if (data == NULL) {
        return;
    }
    mat_free(data->X);
    mat_free(data->y);
    free(data->label);
    free(data);
}

/* ================= 가상 데이터 ================= */

Data *data_make_regression(int n, int d, double noise,
                           double *true_w, double *true_b) {
    Data *data = data_create(n, d, 1, 0);
    if (data == NULL) {
        return NULL;
    }

    /* 진짜 가중치를 무작위로 정한다 (-3 ~ 3) */
    double *w = malloc(sizeof(double) * (size_t)d);
    if (w == NULL) {
        data_free(data);
        return NULL;
    }
    for (int j = 0; j < d; j++) {
        w[j] = ((double)rand() / RAND_MAX) * 6.0 - 3.0;
    }
    double b = ((double)rand() / RAND_MAX) * 4.0 - 2.0;

    for (int i = 0; i < n; i++) {
        double target = b;
        for (int j = 0; j < d; j++) {
            double x = random_normal();          /* 특징도 정규분포로 */
            MAT_AT(data->X, i, j) = x;
            target += w[j] * x;
        }
        /* 잡음을 더한다. 실제 데이터에는 항상 설명되지 않는 변동이 있다 */
        MAT_AT(data->y, i, 0) = target + noise * random_normal();
    }

    if (true_w != NULL) {
        memcpy(true_w, w, sizeof(double) * (size_t)d);
    }
    if (true_b != NULL) {
        *true_b = b;
    }
    free(w);
    return data;
}

Data *data_make_blobs(int n, int d, int k, double spread) {
    Data *data = data_create(n, d, 0, 1);
    if (data == NULL) {
        return NULL;
    }
    data->n_classes = k;

    /* k개의 중심을 -10 ~ 10 범위에 무작위로 놓는다 */
    Matrix *centers = mat_create(k, d);
    if (centers == NULL) {
        data_free(data);
        return NULL;
    }
    for (int c = 0; c < k; c++) {
        for (int j = 0; j < d; j++) {
            MAT_AT(centers, c, j) = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
        }
    }

    for (int i = 0; i < n; i++) {
        int c = i % k;                            /* 클래스를 고르게 배분 */
        for (int j = 0; j < d; j++) {
            MAT_AT(data->X, i, j) =
                MAT_AT(centers, c, j) + spread * random_normal();
        }
        data->label[i] = c;
    }

    mat_free(centers);
    return data;
}

/* ================= 파일 ================= */

int data_save_csv(const Data *data, const char *path) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }

    /* 머리글 */
    for (int j = 0; j < data->d; j++) {
        fprintf(fp, "x%d,", j + 1);
    }
    fprintf(fp, "%s\n", data->label != NULL ? "label" : "y");

    for (int i = 0; i < data->n; i++) {
        for (int j = 0; j < data->d; j++) {
            fprintf(fp, "%.6f,", MAT_AT(data->X, i, j));
        }
        if (data->label != NULL) {
            fprintf(fp, "%d\n", data->label[i]);
        } else {
            fprintf(fp, "%.6f\n", MAT_AT(data->y, i, 0));
        }
    }

    int write_error = ferror(fp);
    if (fclose(fp) != 0 || write_error) {
        return -1;
    }
    return 0;
}

/* 한 줄에 쉼표로 구분된 값이 몇 개인지 센다 */
static int count_columns(const char *line) {
    int count = 1;
    for (const char *p = line; *p != '\0'; p++) {
        if (*p == ',') {
            count++;
        }
    }
    return count;
}

Data *data_load_csv(const char *path, int is_label) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "파일을 열 수 없습니다: %s\n", path);
        return NULL;
    }

    char line[LINE_SIZE];
    if (fgets(line, sizeof(line), fp) == NULL) {    /* 머리글 */
        fclose(fp);
        return NULL;
    }
    int columns = count_columns(line);
    if (columns < 2) {
        fprintf(stderr, "열이 2개 이상이어야 합니다.\n");
        fclose(fp);
        return NULL;
    }

    /* 줄 수를 먼저 센다 (배열 크기를 정하기 위해) */
    int n = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] != '\n' && line[0] != '\r') {
            n++;
        }
    }
    if (n == 0) {
        fprintf(stderr, "데이터가 없습니다.\n");
        fclose(fp);
        return NULL;
    }

    rewind(fp);
    if (fgets(line, sizeof(line), fp) == NULL) {    /* 머리글 건너뛰기 */
        fclose(fp);
        return NULL;
    }

    int d = columns - 1;
    Data *data = data_create(n, d, !is_label, is_label);
    if (data == NULL) {
        fclose(fp);
        return NULL;
    }

    int max_label = 0;
    for (int i = 0; i < n; i++) {
        if (fgets(line, sizeof(line), fp) == NULL) {
            break;
        }
        /* strtok: 문자열을 구분자로 잘라 하나씩 돌려준다 */
        char *token = strtok(line, ",\n\r");
        for (int j = 0; j < d && token != NULL; j++) {
            MAT_AT(data->X, i, j) = atof(token);
            token = strtok(NULL, ",\n\r");
        }
        if (token == NULL) {
            fprintf(stderr, "%d번째 줄의 값이 부족합니다.\n", i + 2);
            data_free(data);
            fclose(fp);
            return NULL;
        }
        if (is_label) {
            data->label[i] = atoi(token);
            if (data->label[i] > max_label) {
                max_label = data->label[i];
            }
        } else {
            MAT_AT(data->y, i, 0) = atof(token);
        }
    }

    if (is_label) {
        data->n_classes = max_label + 1;
    }
    fclose(fp);
    return data;
}

/* ================= 전처리 ================= */

int data_split(const Data *src, double train_ratio,
               Data **train_out, Data **test_out) {
    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        return -1;
    }

    int n_train = (int)(src->n * train_ratio);
    int n_test  = src->n - n_train;
    if (n_train < 1 || n_test < 1) {
        return -1;
    }

    /* 순서를 섞는다. 데이터가 정렬돼 있으면 한쪽에 치우치기 때문 */
    int *order = malloc(sizeof(int) * (size_t)src->n);
    if (order == NULL) {
        return -1;
    }
    for (int i = 0; i < src->n; i++) {
        order[i] = i;
    }
    for (int i = src->n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    int with_y = src->y != NULL;
    int with_label = src->label != NULL;
    Data *train = data_create(n_train, src->d, with_y, with_label);
    Data *test  = data_create(n_test, src->d, with_y, with_label);
    if (train == NULL || test == NULL) {
        free(order);
        data_free(train);
        data_free(test);
        return -1;
    }
    train->n_classes = test->n_classes = src->n_classes;

    for (int i = 0; i < src->n; i++) {
        Data *dst = i < n_train ? train : test;
        int row = i < n_train ? i : i - n_train;
        int from = order[i];

        memcpy(&MAT_AT(dst->X, row, 0), &MAT_AT(src->X, from, 0),
               sizeof(double) * (size_t)src->d);
        if (with_y) {
            MAT_AT(dst->y, row, 0) = MAT_AT(src->y, from, 0);
        }
        if (with_label) {
            dst->label[row] = src->label[from];
        }
    }

    free(order);
    *train_out = train;
    *test_out = test;
    return 0;
}

void data_standardize(Data *train, Data *test) {
    for (int j = 0; j < train->d; j++) {
        double sum = 0.0;
        for (int i = 0; i < train->n; i++) {
            sum += MAT_AT(train->X, i, j);
        }
        double mean = sum / train->n;

        double var = 0.0;
        for (int i = 0; i < train->n; i++) {
            double diff = MAT_AT(train->X, i, j) - mean;
            var += diff * diff;
        }
        double sd = sqrt(var / train->n);
        if (sd < 1e-12) {
            sd = 1.0;                 /* 값이 모두 같은 특징은 그대로 둔다 */
        }

        for (int i = 0; i < train->n; i++) {
            MAT_AT(train->X, i, j) = (MAT_AT(train->X, i, j) - mean) / sd;
        }
        /* 시험 데이터도 "학습 데이터의" 평균과 표준편차로 변환한다.
         * 시험 데이터의 통계를 쓰면 시험 정보가 새어 들어간다(데이터 누수). */
        if (test != NULL) {
            for (int i = 0; i < test->n; i++) {
                MAT_AT(test->X, i, j) = (MAT_AT(test->X, i, j) - mean) / sd;
            }
        }
    }
}

/* ================= 확인 ================= */

void data_summary(const Data *data, const char *name) {
    printf("\n[%s] 샘플 %d개, 특징 %d개", name, data->n, data->d);
    if (data->label != NULL) {
        printf(", 클래스 %d개\n", data->n_classes);
    } else {
        printf(" (회귀)\n");
    }

    printf("특징   평균      표준편차   최소      최대\n");
    for (int j = 0; j < data->d; j++) {
        double sum = 0.0, min = 1e18, max = -1e18;
        for (int i = 0; i < data->n; i++) {
            double v = MAT_AT(data->X, i, j);
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double mean = sum / data->n;

        double var = 0.0;
        for (int i = 0; i < data->n; i++) {
            double diff = MAT_AT(data->X, i, j) - mean;
            var += diff * diff;
        }
        printf("x%-3d %8.3f  %8.3f  %8.3f  %8.3f\n",
               j + 1, mean, sqrt(var / data->n), min, max);
    }

    if (data->y != NULL) {
        double sum = 0.0, min = 1e18, max = -1e18;
        for (int i = 0; i < data->n; i++) {
            double v = MAT_AT(data->y, i, 0);
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        printf("y    %8.3f  %8s  %8.3f  %8.3f\n",
               sum / data->n, "-", min, max);
    }
    if (data->label != NULL) {
        printf("클래스별 개수: ");
        for (int c = 0; c < data->n_classes; c++) {
            int count = 0;
            for (int i = 0; i < data->n; i++) {
                if (data->label[i] == c) count++;
            }
            printf("%d번 %d개  ", c, count);
        }
        printf("\n");
    }
}

#define PLOT_W 60
#define PLOT_H 20

void data_scatter(const Data *data, int fx, int fy) {
    if (fx < 0 || fx >= data->d || fy < 0 || fy >= data->d) {
        printf("특징 번호가 범위를 벗어났습니다.\n");
        return;
    }

    /* 그릴 범위를 먼저 구한다 */
    double min_x = 1e18, max_x = -1e18, min_y = 1e18, max_y = -1e18;
    for (int i = 0; i < data->n; i++) {
        double x = MAT_AT(data->X, i, fx);
        double y = MAT_AT(data->X, i, fy);
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    if (range_x < 1e-12) range_x = 1.0;
    if (range_y < 1e-12) range_y = 1.0;

    char canvas[PLOT_H][PLOT_W + 1];
    memset(canvas, ' ', sizeof(canvas));
    for (int r = 0; r < PLOT_H; r++) {
        canvas[r][PLOT_W] = '\0';
    }

    static const char *marks = "ABCDEFGHIJ";
    for (int i = 0; i < data->n; i++) {
        int col = (int)((MAT_AT(data->X, i, fx) - min_x) / range_x * (PLOT_W - 1));
        int row = (int)((MAT_AT(data->X, i, fy) - min_y) / range_y * (PLOT_H - 1));
        row = PLOT_H - 1 - row;                 /* 화면은 위가 0번 줄 */

        char mark = '*';
        if (data->label != NULL) {
            mark = marks[data->label[i] % 10];  /* 클래스마다 다른 글자 */
        }
        canvas[row][col] = mark;
    }

    printf("\n산점도: 가로 x%d (%.2f ~ %.2f), 세로 x%d (%.2f ~ %.2f)\n",
           fx + 1, min_x, max_x, fy + 1, min_y, max_y);
    for (int r = 0; r < PLOT_H; r++) {
        printf("|%s|\n", canvas[r]);
    }
    if (data->label != NULL) {
        printf("클래스: ");
        for (int c = 0; c < data->n_classes && c < 10; c++) {
            printf("%c=%d번  ", marks[c], c);
        }
        printf("\n");
    }
}
