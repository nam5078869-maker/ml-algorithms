#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

static int same_shape(const Matrix *a, const Matrix *b) {
    return a->rows == b->rows && a->cols == b->cols;
}

static int mat_size(const Matrix *m) {
    return m->rows * m->cols;
}

Matrix *mat_create(int rows, int cols) {
    assert(rows > 0 && cols > 0);

    Matrix *m = malloc(sizeof(Matrix));
    if (m == NULL) {
        return NULL;
    }
    m->data = calloc((size_t)rows * (size_t)cols, sizeof(double));
    if (m->data == NULL) {
        free(m);
        return NULL;
    }
    m->rows = rows;
    m->cols = cols;
    return m;
}

void mat_free(Matrix *m) {
    if (m == NULL) {
        return;
    }
    free(m->data);
    free(m);
}

void mat_set_values(Matrix *m, const double *values) {
    memcpy(m->data, values, sizeof(double) * (size_t)mat_size(m));
}

void mat_fill(Matrix *m, double value) {
    for (int i = 0; i < mat_size(m); i++) {
        m->data[i] = value;
    }
}

void mat_randomize(Matrix *m, double limit) {
    for (int i = 0; i < mat_size(m); i++) {
        double r = (double)rand() / RAND_MAX;
        m->data[i] = (r * 2.0 - 1.0) * limit;
    }
}

void mat_copy(Matrix *dst, const Matrix *src) {
    assert(same_shape(dst, src));
    memcpy(dst->data, src->data, sizeof(double) * (size_t)mat_size(src));
}

void mat_mul(Matrix *out, const Matrix *a, const Matrix *b) {
    assert(a->cols == b->rows);
    assert(out->rows == a->rows && out->cols == b->cols);
    assert(out != a && out != b);

    mat_fill(out, 0.0);

    /* i-k-j 순서: 안쪽 반복에서 메모리를 순서대로 읽어 캐시 효율이 좋다 */
    for (int i = 0; i < a->rows; i++) {
        for (int k = 0; k < a->cols; k++) {
            double a_ik = MAT_AT(a, i, k);
            for (int j = 0; j < b->cols; j++) {
                MAT_AT(out, i, j) += a_ik * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_add(Matrix *out, const Matrix *a, const Matrix *b) {
    assert(same_shape(a, b) && same_shape(out, a));
    for (int i = 0; i < mat_size(a); i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void mat_sub(Matrix *out, const Matrix *a, const Matrix *b) {
    assert(same_shape(a, b) && same_shape(out, a));
    for (int i = 0; i < mat_size(a); i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void mat_hadamard(Matrix *out, const Matrix *a, const Matrix *b) {
    assert(same_shape(a, b) && same_shape(out, a));
    for (int i = 0; i < mat_size(a); i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

void mat_scale(Matrix *out, const Matrix *a, double k) {
    assert(same_shape(out, a));
    for (int i = 0; i < mat_size(a); i++) {
        out->data[i] = a->data[i] * k;
    }
}

void mat_transpose(Matrix *out, const Matrix *a) {
    assert(out->rows == a->cols && out->cols == a->rows);
    assert(out != a);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            MAT_AT(out, j, i) = MAT_AT(a, i, j);
        }
    }
}

void mat_apply(Matrix *out, const Matrix *a, double (*f)(double)) {
    assert(same_shape(out, a));
    for (int i = 0; i < mat_size(a); i++) {
        out->data[i] = f(a->data[i]);
    }
}

int mat_equals(const Matrix *a, const Matrix *b, double eps) {
    if (!same_shape(a, b)) {
        return 0;
    }
    for (int i = 0; i < mat_size(a); i++) {
        if (fabs(a->data[i] - b->data[i]) > eps) {
            return 0;
        }
    }
    return 1;
}

void mat_print(const Matrix *m, const char *name) {
    printf("%s (%dx%d) = [\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  ");
        for (int j = 0; j < m->cols; j++) {
            printf("%8.3f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}
