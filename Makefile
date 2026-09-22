CC      = clang
CFLAGS  = -Wall -Wextra -std=c11 -O2 -g
LDLIBS  = -lm

LIB_OBJS = matrix.o dataset.o metrics.o linreg.o knn.o mnist.o kmeans.o
PROGRAMS = gen_data linreg_demo knn_demo kmeans_demo

all: $(PROGRAMS)

gen_data: gen_data.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

linreg_demo: linreg_demo.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

knn_demo: knn_demo.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

kmeans_demo: kmeans_demo.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c matrix.h dataset.h metrics.h linreg.h knn.h mnist.h kmeans.h
	$(CC) $(CFLAGS) -c $<

# 1단계: 데이터 생성과 확인
run-data: gen_data
	./gen_data

# 2단계: 선형 회귀
run-linreg: linreg_demo
	./linreg_demo

# 3단계: KNN (MNIST는 ../neural-net/data 사용)
run-knn: knn_demo
	./knn_demo

# 4단계: k-means
run-kmeans: kmeans_demo
	./kmeans_demo

# 메모리 오류 검사를 켜고 빌드
debug: CFLAGS := -Wall -Wextra -std=c11 -g -fsanitize=address,undefined
debug: clean all

clean:
	rm -f *.o $(PROGRAMS)

.PHONY: all run-data run-linreg run-knn run-kmeans debug clean
