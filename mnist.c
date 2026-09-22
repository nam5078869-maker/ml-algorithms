/*
 * MNIST 데이터 파일(IDX 형식) 읽기
 *
 * 파일 구조 (모든 정수는 빅 엔디안 32비트):
 *
 *   이미지 파일            라벨 파일
 *   0000  매직 넘버 2051   0000  매직 넘버 2049
 *   0004  이미지 개수      0004  라벨 개수
 *   0008  행 수 (28)       0008~ 라벨 1바이트씩
 *   0012  열 수 (28)
 *   0016~ 픽셀 1바이트씩 (0=흰색, 255=검은색)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist.h"

/*
 * 빅 엔디안 4바이트를 읽어 int로 바꾼다.
 *
 * 파일에는 큰 자리 바이트가 먼저 저장돼 있는데(빅 엔디안),
 * 맥과 PC의 CPU는 작은 자리를 먼저 쓴다(리틀 엔디안).
 * 그래서 4바이트를 그냥 읽으면 값이 뒤집힌다.
 * 한 바이트씩 읽어 자리를 직접 맞추면 어떤 컴퓨터에서도 똑같이 동작한다.
 */
static int read_be_int(FILE *fp, int *out) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, fp) != 4) {
        return 0;
    }
    *out = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
    return 1;
}

void mnist_free(Dataset *ds) {
    if (ds == NULL) {
        return;
    }
    mat_free(ds->images);
    mat_free(ds->labels);
    free(ds->digits);
    free(ds);
}

Dataset *mnist_load(const char *image_path, const char *label_path,
                    int max_count) {
    FILE *fi = fopen(image_path, "rb");     /* "rb": 바이너리 읽기 모드 */
    if (fi == NULL) {
        fprintf(stderr, "이미지 파일을 열 수 없습니다: %s\n", image_path);
        return NULL;
    }
    FILE *fl = fopen(label_path, "rb");
    if (fl == NULL) {
        fprintf(stderr, "라벨 파일을 열 수 없습니다: %s\n", label_path);
        fclose(fi);
        return NULL;
    }

    Dataset *ds = NULL;
    unsigned char *buffer = NULL;

    /* ---- 머리말 읽기 ---- */
    int magic, image_count, rows, cols;
    int label_magic, label_count;

    if (!read_be_int(fi, &magic) || !read_be_int(fi, &image_count)
        || !read_be_int(fi, &rows) || !read_be_int(fi, &cols)
        || !read_be_int(fl, &label_magic) || !read_be_int(fl, &label_count)) {
        fprintf(stderr, "파일 머리말을 읽을 수 없습니다.\n");
        goto cleanup;
    }
    if (magic != 2051 || label_magic != 2049) {
        fprintf(stderr, "MNIST 파일이 아닙니다 (매직 넘버 %d, %d).\n",
                magic, label_magic);
        goto cleanup;
    }
    if (image_count != label_count) {
        fprintf(stderr, "이미지 수(%d)와 라벨 수(%d)가 다릅니다.\n",
                image_count, label_count);
        goto cleanup;
    }
    if (rows * cols != MNIST_IMAGE_SIZE) {
        fprintf(stderr, "이미지 크기가 28x28이 아닙니다 (%dx%d).\n", rows, cols);
        goto cleanup;
    }

    int count = image_count;
    if (max_count > 0 && max_count < count) {
        count = max_count;
    }

    /* ---- 메모리 준비 ---- */
    ds = calloc(1, sizeof(Dataset));
    if (ds == NULL) {
        goto cleanup;
    }
    ds->count  = count;
    ds->rows   = rows;
    ds->cols   = cols;
    ds->images = mat_create(count, MNIST_IMAGE_SIZE);
    ds->labels = mat_create(count, MNIST_CLASSES);
    ds->digits = malloc((size_t)count);
    buffer     = malloc(MNIST_IMAGE_SIZE);

    if (ds->images == NULL || ds->labels == NULL
        || ds->digits == NULL || buffer == NULL) {
        fprintf(stderr, "메모리가 부족합니다.\n");
        goto cleanup;
    }

    /* ---- 이미지와 라벨 읽기 ---- */
    for (int i = 0; i < count; i++) {
        if (fread(buffer, 1, MNIST_IMAGE_SIZE, fi) != MNIST_IMAGE_SIZE) {
            fprintf(stderr, "%d번째 이미지를 읽지 못했습니다.\n", i);
            goto cleanup;
        }

        /* 픽셀 0~255를 0.0~1.0으로 바꾼다 (정규화).
         * 입력 값이 크면 가중치 합이 커져 학습이 불안정해진다. */
        for (int p = 0; p < MNIST_IMAGE_SIZE; p++) {
            MAT_AT(ds->images, i, p) = buffer[p] / 255.0;
        }

        int digit = fgetc(fl);
        if (digit < 0 || digit > 9) {
            fprintf(stderr, "%d번째 라벨이 이상합니다 (%d).\n", i, digit);
            goto cleanup;
        }
        ds->digits[i] = (unsigned char)digit;

        /* 원-핫 벡터: 정답 자리만 1, 나머지는 0
         * 예) 3 → [0 0 0 1 0 0 0 0 0 0] */
        MAT_AT(ds->labels, i, digit) = 1.0;
    }

    free(buffer);
    fclose(fi);
    fclose(fl);
    return ds;

cleanup:
    free(buffer);
    mnist_free(ds);
    fclose(fi);
    fclose(fl);
    return NULL;
}

/* 픽셀 밝기를 문자로 바꾼다 (진할수록 뒤쪽 문자) */
static char shade(double value) {
    static const char *levels = " .:-=+*#%@";
    int index = (int)(value * 9.99);
    if (index < 0) index = 0;
    if (index > 9) index = 9;
    return levels[index];
}

void mnist_print_image(const Dataset *ds, int index) {
    if (index < 0 || index >= ds->count) {
        printf("범위를 벗어난 번호입니다.\n");
        return;
    }

    printf("[%d번째 이미지] 정답: %d\n", index, ds->digits[index]);
    for (int r = 0; r < ds->rows; r++) {
        for (int c = 0; c < ds->cols; c++) {
            double v = MAT_AT(ds->images, index, r * ds->cols + c);
            putchar(shade(v));
            putchar(shade(v));      /* 가로로 2칸씩 그려야 정사각형으로 보인다 */
        }
        putchar('\n');
    }
}
