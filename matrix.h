#ifndef MATRIX_H
#define MATRIX_H

/*
 * 2차원 행렬을 1차원 배열 하나에 행 우선(row-major)으로 저장한다.
 *   (i행, j열) 값 = data[i * cols + j]
 *
 * (neural-net 프로젝트에서 만든 것을 그대로 사용)
 */

typedef struct {
    int     rows;
    int     cols;
    double *data;
} Matrix;

/* (i, j) 원소에 접근하는 매크로. 읽기와 쓰기 모두 가능 */
#define MAT_AT(m, i, j) ((m)->data[(i) * (m)->cols + (j)])

Matrix *mat_create(int rows, int cols);
void    mat_free(Matrix *m);
void    mat_set_values(Matrix *m, const double *values);
void    mat_fill(Matrix *m, double value);
void    mat_randomize(Matrix *m, double limit);
void    mat_copy(Matrix *dst, const Matrix *src);

void mat_mul(Matrix *out, const Matrix *a, const Matrix *b);
void mat_add(Matrix *out, const Matrix *a, const Matrix *b);
void mat_sub(Matrix *out, const Matrix *a, const Matrix *b);
void mat_hadamard(Matrix *out, const Matrix *a, const Matrix *b);
void mat_scale(Matrix *out, const Matrix *a, double k);
void mat_transpose(Matrix *out, const Matrix *a);
void mat_apply(Matrix *out, const Matrix *a, double (*f)(double));

int  mat_equals(const Matrix *a, const Matrix *b, double eps);
void mat_print(const Matrix *m, const char *name);

#endif
