/*
 * 3단계: KNN 분류
 *
 *   실험 1  k에 따라 결정 경계가 어떻게 바뀌는지 그림으로 보기
 *   실험 2  교차검증으로 k 고르기
 *   실험 3  MNIST 손글씨에 적용해서 신경망(97.76%)과 비교
 *
 * 실행: ./knn_demo [MNIST 학습 장수] [MNIST 시험 장수]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dataset.h"
#include "knn.h"
#include "metrics.h"
#include "mnist.h"

#define SEED 42

/*
 * MNIST 폴더 찾기: 이 프로젝트의 data 폴더가 있으면 그것을,
 * 없으면 neural-net 프로젝트의 data 폴더를 쓴다.
 */
static const char *mnist_dir(void) {
    static const char *candidates[] = {"data", "../neural-net/data"};
    for (int i = 0; i < 2; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", candidates[i]);
        FILE *fp = fopen(path, "rb");
        if (fp != NULL) {
            fclose(fp);
            return candidates[i];
        }
    }
    return "data";
}

/* 폴더와 파일 이름을 이어 경로를 만든다 */
static const char *mnist_path(char *buf, size_t size, const char *file) {
    snprintf(buf, size, "%s/%s", mnist_dir(), file);
    return buf;
}

static double seconds_since(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

/* ---------- 결정 경계 그리기 ---------- */

#define GW 60
#define GH 20

/*
 * 화면의 모든 칸에 대해 "여기에 점이 있다면 무슨 클래스일까"를 예측해
 * 소문자로 칠하고, 실제 학습 데이터는 대문자로 겹쳐 그린다.
 */
static void plot_boundary(const KNN *model, const Data *data) {
    double min_x = 1e18, max_x = -1e18, min_y = 1e18, max_y = -1e18;
    for (int i = 0; i < data->n; i++) {
        double x = MAT_AT(data->X, i, 0), y = MAT_AT(data->X, i, 1);
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }

    static const char *lower = "abcdefghij";
    static const char *upper = "ABCDEFGHIJ";
    char canvas[GH][GW + 1];

    for (int r = 0; r < GH; r++) {
        for (int c = 0; c < GW; c++) {
            double point[2] = {
                min_x + (max_x - min_x) * c / (GW - 1),
                max_y - (max_y - min_y) * r / (GH - 1),
            };
            canvas[r][c] = lower[knn_predict_one(model, point)];
        }
        canvas[r][GW] = '\0';
    }
    for (int i = 0; i < data->n; i++) {
        int c = (int)((MAT_AT(data->X, i, 0) - min_x) / (max_x - min_x) * (GW - 1));
        int r = (int)((max_y - MAT_AT(data->X, i, 1)) / (max_y - min_y) * (GH - 1));
        canvas[r][c] = upper[data->label[i]];
    }
    for (int r = 0; r < GH; r++) {
        printf("|%s|\n", canvas[r]);
    }
}

static void experiment_boundary(void) {
    printf("==================== 실험 1: k와 결정 경계 ====================\n");
    printf("대문자 = 학습 데이터, 소문자 = 그 위치의 예측 결과\n");

    /* 군집이 조금 섞이도록 spread를 크게 */
    Data *all = data_make_blobs(300, 2, 3, 2.5);
    Data *train, *test;
    data_split(all, 0.7, &train, &test);

    int ks[] = {1, 15};
    for (int i = 0; i < 2; i++) {
        KNN model;
        knn_init(&model, train->X, train->label, train->n_classes, ks[i]);

        int *pred = malloc(sizeof(int) * (size_t)test->n);
        knn_predict(&model, test->X, pred);
        printf("\n[k = %d]  시험 정확도 %.1f%%\n", ks[i],
               metric_accuracy(pred, test->label, test->n));
        free(pred);

        plot_boundary(&model, train);
    }

    printf("\nk가 작으면 경계가 들쭉날쭉하고(학습 데이터에 과하게 맞춤),\n"
           "k가 크면 경계가 매끄러워진다.\n");

    data_free(all);
    data_free(train);
    data_free(test);
}

/* ---------- 교차검증으로 k 고르기 ---------- */

static void experiment_choose_k(void) {
    printf("\n==================== 실험 2: 교차검증으로 k 고르기 ====================\n");
    Data *all = data_make_blobs(600, 2, 4, 2.5);
    Data *train, *test;
    data_split(all, 0.8, &train, &test);

    int ks[] = {1, 3, 5, 9, 15, 31, 61, 121};
    int n_ks = (int)(sizeof(ks) / sizeof(ks[0]));
    int best_k = ks[0];
    double best_acc = -1.0;

    printf("   k   5겹 교차검증 정확도\n");
    for (int i = 0; i < n_ks; i++) {
        double acc = knn_cross_validate(train->X, train->label,
                                        train->n_classes, ks[i], 5);
        printf("%4d   %6.2f%%  ", ks[i], acc);
        int bars = (int)((acc - 70.0) * 1.5);
        for (int b = 0; b < bars; b++) putchar('#');
        putchar('\n');
        if (acc > best_acc) {
            best_acc = acc;
            best_k = ks[i];
        }
    }

    KNN model;
    knn_init(&model, train->X, train->label, train->n_classes, best_k);
    int *pred = malloc(sizeof(int) * (size_t)test->n);
    knn_predict(&model, test->X, pred);
    printf("\n교차검증으로 고른 k = %d → 시험 정확도 %.2f%%\n",
           best_k, metric_accuracy(pred, test->label, test->n));
    printf("(시험 데이터는 k를 고르는 동안 한 번도 사용하지 않았다)\n");

    free(pred);
    data_free(all);
    data_free(train);
    data_free(test);
}

/* ---------- MNIST ---------- */

static void experiment_mnist(int n_train, int n_test) {
    char p1[256], p2[256], p3[256], p4[256];
    printf("\n==================== 실험 3: MNIST ====================\n");
    Dataset *train = mnist_load(mnist_path(p1, sizeof(p1), "train-images-idx3-ubyte"),
                                mnist_path(p2, sizeof(p2), "train-labels-idx1-ubyte"), n_train);
    Dataset *test  = mnist_load(mnist_path(p3, sizeof(p3), "t10k-images-idx3-ubyte"),
                                mnist_path(p4, sizeof(p4), "t10k-labels-idx1-ubyte"), n_test);
    if (train == NULL || test == NULL) {
        printf("MNIST 데이터가 없습니다. data 폴더에 파일을 넣어 주세요 (README 참고).\n");
        mnist_free(train);
        mnist_free(test);
        return;
    }

    /* Dataset의 digits(unsigned char)를 int 라벨로 바꾼다 */
    int *train_label = malloc(sizeof(int) * (size_t)train->count);
    int *test_label  = malloc(sizeof(int) * (size_t)test->count);
    int *pred        = malloc(sizeof(int) * (size_t)test->count);
    for (int i = 0; i < train->count; i++) train_label[i] = train->digits[i];
    for (int i = 0; i < test->count; i++)  test_label[i]  = test->digits[i];

    printf("학습 %d장을 기억해 두고, 시험 %d장마다 가장 가까운 이웃을 찾는다.\n",
           train->count, test->count);
    printf("비교 횟수: %d x %d = 약 %.1f천만 번 (각각 784개 픽셀)\n\n",
           train->count, test->count,
           (double)train->count * test->count / 1e7);

    int ks[] = {1, 3, 5};
    for (int i = 0; i < 3; i++) {
        KNN model;
        knn_init(&model, train->images, train_label, 10, ks[i]);

        clock_t t = clock();
        knn_predict(&model, test->images, pred);
        double sec = seconds_since(t);

        printf("k = %d   정확도 %.2f%%   예측 시간 %.1f초 (한 장당 %.1fms)\n",
               ks[i], metric_accuracy(pred, test_label, test->count),
               sec, sec * 1000.0 / test->count);
    }

    printf("\n참고: 신경망(neural-net)은 6만 장으로 학습해 97.76%%,\n"
           "      학습 후 예측은 한 장당 1ms도 걸리지 않는다.\n");

    free(train_label);
    free(test_label);
    free(pred);
    mnist_free(train);
    mnist_free(test);
}

int main(int argc, char **argv) {
    int n_train = argc > 1 ? atoi(argv[1]) : 10000;
    int n_test  = argc > 2 ? atoi(argv[2]) : 1000;

    srand(SEED);
    experiment_boundary();
    experiment_choose_k();
    experiment_mnist(n_train, n_test);
    return 0;
}
