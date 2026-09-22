#ifndef MNIST_H
#define MNIST_H

#include "matrix.h"

#define MNIST_IMAGE_SIZE 784   /* 28 x 28 */
#define MNIST_CLASSES    10    /* 숫자 0~9 */

/* MNIST 데이터 묶음 */
typedef struct {
    int count;              /* 이미지 개수 */
    int rows, cols;         /* 28, 28 */

    Matrix *images;         /* count x 784, 각 값은 0.0 ~ 1.0 */
    Matrix *labels;         /* count x 10, 정답만 1인 원-핫 벡터 */
    unsigned char *digits;  /* count개, 정답 숫자 0~9 (출력·채점용) */
} Dataset;

/*
 * IDX 형식 파일에서 이미지와 라벨을 읽는다.
 * max_count가 0보다 크면 그 개수만 읽는다 (메모리 절약용).
 * 실패하면 NULL을 돌려주고 이유를 stderr에 출력한다.
 */
Dataset *mnist_load(const char *image_path, const char *label_path,
                    int max_count);

void mnist_free(Dataset *ds);

/* index번째 이미지를 터미널에 문자로 그린다 */
void mnist_print_image(const Dataset *ds, int index);

#endif
