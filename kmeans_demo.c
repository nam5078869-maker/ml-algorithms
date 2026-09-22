/*
 * 4단계: k-means 군집화
 *
 *   실험 1  반복하면서 중심이 움직이는 모습 보기
 *   실험 2  무작위 초기화 vs k-means++
 *   실험 3  엘보우 방법으로 k 고르기
 *   실험 4  MNIST를 정답 없이 10개로 묶어 보기
 *
 * 실행: ./kmeans_demo [MNIST 장수]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataset.h"
#include "kmeans.h"
#include "mnist.h"

#define SEED 42

/* ---------- 군집 그림 ---------- */

#define GW 60
#define GH 18

/* 군집별로 다른 글자(a, b, c...)로 점을 찍고, 중심은 숫자(0, 1, 2...)로 */
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

static void plot_clusters(const KMeans *m, const Matrix *X) {
    double min_x = 1e18, max_x = -1e18, min_y = 1e18, max_y = -1e18;
    for (int i = 0; i < X->rows; i++) {
        double x = MAT_AT(X, i, 0), y = MAT_AT(X, i, 1);
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }

    char canvas[GH][GW + 1];
    memset(canvas, ' ', sizeof(canvas));
    for (int r = 0; r < GH; r++) canvas[r][GW] = '\0';

    static const char *marks = "abcdefghij";
    for (int i = 0; i < X->rows; i++) {
        int c = (int)((MAT_AT(X, i, 0) - min_x) / (max_x - min_x) * (GW - 1));
        int r = (int)((max_y - MAT_AT(X, i, 1)) / (max_y - min_y) * (GH - 1));
        canvas[r][c] = m->assign[i] < 0 ? '.' : marks[m->assign[i] % 10];
    }
    for (int k = 0; k < m->k; k++) {
        int c = (int)((MAT_AT(m->centroids, k, 0) - min_x) / (max_x - min_x) * (GW - 1));
        int r = (int)((max_y - MAT_AT(m->centroids, k, 1)) / (max_y - min_y) * (GH - 1));
        if (c >= 0 && c < GW && r >= 0 && r < GH) {
            canvas[r][c] = (char)('0' + k);
        }
    }
    for (int r = 0; r < GH; r++) printf("|%s|\n", canvas[r]);
}

static void experiment_steps(void) {
    printf("==================== 실험 1: 반복 과정 ====================\n");
    printf("소문자 = 점이 배정된 군집, 숫자 = 군집의 중심\n");

    Data *data = data_make_blobs(300, 2, 4, 1.3);
    KMeans *m = kmeans_create(4, 2, data->n);

    /* 일부러 무작위 초기화로 시작해 중심이 움직이는 모습을 본다 */
    kmeans_init_centroids(m, data->X, INIT_RANDOM);
    printf("\n[시작] 무작위로 고른 중심 4개\n");
    plot_clusters(m, data->X);

    for (int iter = 1; iter <= 20; iter++) {
        int changed = kmeans_assign_step(m, data->X);
        if (iter == 1 || changed == 0) {
            printf("\n[%d번째 배정] 바뀐 점 %d개, 관성 %.1f%s\n", iter, changed,
                   m->inertia, changed == 0 ? "  → 수렴!" : "");
            plot_clusters(m, data->X);
        }
        if (changed == 0) {
            break;
        }
        printf("  %d번째 반복: 바뀐 점 %3d개, 관성 %.1f\n", iter, changed, m->inertia);
        kmeans_update_step(m, data->X);
    }

    kmeans_free(m);
    data_free(data);
}

/* ---------- 초기화 비교 ---------- */

static void experiment_init(void) {
    printf("\n==================== 실험 2: 초기화 방법 비교 ====================\n");
    Data *data = data_make_blobs(600, 2, 8, 1.0);
    KMeans *m = kmeans_create(8, 2, data->n);

    const int runs = 30;
    /* 한글은 화면에서 2칸이라 %-10s로 맞추면 어긋나므로 공백을 직접 넣는다 */
    const char *names[] = {"무작위    ", "k-means++ "};
    KMeansInit inits[] = {INIT_RANDOM, INIT_PLUSPLUS};

    /* 같은 데이터로 30번씩 실행해, 가장 좋은 관성 대비 얼마나 나빴는지 본다 */
    double results[2][30];
    int    iters[2][30];
    double best_overall = 1e18;
    for (int t = 0; t < 2; t++) {
        for (int r = 0; r < runs; r++) {
            iters[t][r] = kmeans_fit(m, data->X, inits[t], 300);
            results[t][r] = m->inertia;
            if (m->inertia < best_overall) best_overall = m->inertia;
        }
    }

    printf("군집 8개, 같은 데이터로 %d번씩 실행\n\n", runs);
    printf("방법        평균 관성   최악 관성   최선 대비 5%% 이내   평균 반복\n");
    for (int t = 0; t < 2; t++) {
        double sum = 0.0, worst = 0.0, it = 0.0;
        int good = 0;
        for (int r = 0; r < runs; r++) {
            sum += results[t][r];
            it += iters[t][r];
            if (results[t][r] > worst) worst = results[t][r];
            if (results[t][r] <= best_overall * 1.05) good++;
        }
        printf("%s %9.1f  %9.1f   %2d / %d번          %5.1f\n",
               names[t], sum / runs, worst, good, runs, it / runs);
    }
    printf("\nk-means는 시작점에 따라 나쁜 답(지역 최솟값)에 빠질 수 있다.\n"
           "k-means++는 중심을 흩어서 시작하므로 이런 일이 훨씬 적다.\n");

    kmeans_free(m);
    data_free(data);
}

/* ---------- 엘보우 ---------- */

static void experiment_elbow(void) {
    printf("\n==================== 실험 3: 엘보우 방법 ====================\n");
    printf("실제 군집 수가 5개인 데이터에서 k를 1~10으로 바꿔 본다.\n\n");
    Data *data = data_make_blobs(500, 2, 5, 1.0);

    double inertia[11];
    for (int k = 1; k <= 10; k++) {
        /* 초기값 운을 줄이려고 5번 중 가장 좋은 결과를 쓴다 */
        double best = 1e18;
        KMeans *m = kmeans_create(k, 2, data->n);
        for (int r = 0; r < 5; r++) {
            kmeans_fit(m, data->X, INIT_PLUSPLUS, 300);
            if (m->inertia < best) best = m->inertia;
        }
        inertia[k] = best;
        kmeans_free(m);
    }

    printf(" k    관성\n");
    for (int k = 1; k <= 10; k++) {
        printf("%2d  %8.1f  ", k, inertia[k]);
        int bars = (int)(inertia[k] / inertia[1] * 50);
        for (int b = 0; b < bars; b++) putchar('#');
        if (k > 1) {
            printf("  (-%.0f%%)", 100.0 * (1.0 - inertia[k] / inertia[k - 1]));
        }
        putchar('\n');
    }
    printf("\nk가 커질수록 관성은 항상 줄어든다 (k = 점 개수면 0).\n"
           "그래서 가장 작은 값이 아니라 \"줄어드는 폭이 갑자기 작아지는 지점\"\n"
           "(팔꿈치)을 고른다.\n");

    data_free(data);
}

/* ---------- MNIST ---------- */

static char shade(double v) {
    static const char *levels = " .:-=+*#%@";
    int i = (int)(v * 9.99);
    if (i < 0) i = 0;
    if (i > 9) i = 9;
    return levels[i];
}

/* 중심 여러 개를 가로로 나란히 그린다 (각 14x14로 줄여서) */
static void print_centroids(const KMeans *m, const int *order, int count,
                            const int *major) {
    for (int c = 0; c < count; c++) {
        printf("  군집%-2d(%d) ", order[c], major[order[c]]);
    }
    printf("\n");
    for (int r = 0; r < 28; r += 2) {
        for (int c = 0; c < count; c++) {
            printf("  ");
            for (int col = 0; col < 28; col += 2) {
                /* 2x2 픽셀을 평균 내서 한 칸으로 */
                const double *img = &MAT_AT(m->centroids, order[c], 0);
                double v = (img[r * 28 + col] + img[r * 28 + col + 1]
                          + img[(r + 1) * 28 + col] + img[(r + 1) * 28 + col + 1]) / 4.0;
                putchar(shade(v));
            }
            printf("  ");
        }
        printf("\n");
    }
}

static void experiment_mnist(int n) {
    char p1[256], p2[256];
    printf("\n==================== 실험 4: MNIST ====================\n");
    Dataset *ds = mnist_load(mnist_path(p1, sizeof(p1), "train-images-idx3-ubyte"),
                             mnist_path(p2, sizeof(p2), "train-labels-idx1-ubyte"), n);
    if (ds == NULL) {
        printf("MNIST 데이터가 없습니다. data 폴더에 파일을 넣어 주세요 (README 참고).\n");
        return;
    }

    printf("손글씨 %d장을 정답을 보지 않고 10개 군집으로 묶는다.\n", ds->count);
    KMeans *m = kmeans_create(10, MNIST_IMAGE_SIZE, ds->count);
    int iters = kmeans_fit(m, ds->images, INIT_PLUSPLUS, 100);
    printf("%d번 반복 후 수렴\n\n", iters);

    /* 끝난 뒤에야 정답을 꺼내서, 각 군집에 어떤 숫자가 모였는지 센다 */
    int table[10][10] = {{0}};
    int size[10] = {0};
    for (int i = 0; i < ds->count; i++) {
        table[m->assign[i]][ds->digits[i]]++;
        size[m->assign[i]]++;
    }

    int major[10];
    int total_major = 0;
    printf("군집   크기   가장 많은 숫자   순도   숫자별 개수 (0~9)\n");
    for (int c = 0; c < 10; c++) {
        int best = 0;
        for (int dgt = 1; dgt < 10; dgt++) {
            if (table[c][dgt] > table[c][best]) best = dgt;
        }
        major[c] = best;
        total_major += table[c][best];
        printf("%3d   %5d        %d        %5.1f%%  ", c, size[c], best,
               size[c] ? 100.0 * table[c][best] / size[c] : 0.0);
        for (int dgt = 0; dgt < 10; dgt++) {
            printf("%4d", table[c][dgt]);
        }
        printf("\n");
    }
    printf("\n전체 순도 %.1f%%: 각 군집을 가장 많은 숫자로 불렀을 때 맞는 비율\n",
           100.0 * total_major / ds->count);

    int missing[10] = {0};
    for (int c = 0; c < 10; c++) missing[major[c]] = 1;
    printf("군집의 대표가 되지 못한 숫자: ");
    int any = 0;
    for (int dgt = 0; dgt < 10; dgt++) {
        if (!missing[dgt]) { printf("%d ", dgt); any = 1; }
    }
    printf("%s\n", any ? "" : "없음");

    printf("\n[군집 중심 = 그 군집의 평균 이미지] 괄호 안은 가장 많은 숫자\n");
    int order[10];
    for (int c = 0; c < 10; c++) order[c] = c;
    print_centroids(m, order, 5, major);
    print_centroids(m, order + 5, 5, major);

    kmeans_free(m);
    mnist_free(ds);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 5000;
    srand(SEED);
    experiment_steps();
    experiment_init();
    experiment_elbow();
    experiment_mnist(n);
    return 0;
}
