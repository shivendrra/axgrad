#include <stdlib.h>

int randint(int low, int high) {
    return low + rand() % (high - low + 1);
}

int* randint_array(int low, int high, int size) {
    int* array = (int*)malloc(size * sizeof(int));
    if (array == NULL) {
        return NULL;
    }
    for (int i = 0; i < size; ++i) {
        array[i] = randint(low, high);
    }
    return array;
}

int* zeros(int* shape, int ndim) {
    int* array;
    if (ndim == 1) {
        array = (int*)malloc(shape[0] * sizeof(int));
        if (array == NULL) {
            return NULL;
        }
        for (int i = 0; i < shape[0]; ++i) {
            array[i] = 0;
        }
    } else {
        array = (int*)malloc(shape[0] * sizeof(int*));
        if (array == NULL) {
            return NULL;
        }
        for (int i = 0; i < shape[0]; ++i) {
            array[i] = zeros(shape + 1, ndim - 1);
            if (array[i] == NULL) {
                for (int j = 0; j < i; ++j) {
                    free(array[j]);
                }
                free(array);
                return NULL;
            }
        }
    }
    return array;
}

int* ones(int* shape, int ndim) {
    int* array;
    if (ndim == 1) {
        array = (int*)malloc(shape[0] * sizeof(int));
        if (array == NULL) {
            return NULL;
        }
        for (int i = 0; i < shape[0]; ++i) {
            array[i] = 1;
        }
    } else {
        array = (int*)malloc(shape[0] * sizeof(int*));
        if (array == NULL) {
            return NULL;
        }
        for (int i = 0; i < shape[0]; ++i) {
            array[i] = ones(shape + 1, ndim - 1);
            if (array[i] == NULL) {
                for (int j = 0; j < i; ++j) {
                    free(array[j]);
                }
                free(array);
                return NULL;
            }
        }
    }
    return array;
}

int* ns(int* shape, int ndim, int n) {
    int* array;
    if (ndim == 1) {
        array = (int*)malloc(shape[0] * sizeof(int));
        if (array == NULL) {
            return NULL;
        }
        for (int i = 0; i < shape[0]; ++i) {
            array[i] = n;
        }
    } else {
        array = (int*)malloc(shape[0] * sizeof(int*));
        if (array == NULL) {
            return NULL;
        }
        for (int i = 0; i < shape[0]; ++i) {
            array[i] = ns(shape + 1, ndim - 1, n);
            if (array[i] == NULL) {
                
                for (int j = 0; j < i; ++j) {
                    free(array[j]);
                }
                free(array);
                return NULL;
            }
        }
    }
    return array;
}

double* arange(double start, double end, double step) {
    int size = (int)((end - start) / step);
    double* array = (double*)malloc(size * sizeof(double));
    if (array == NULL) {
        return NULL;
    }
    for (int i = 0; i < size; ++i) {
        array[i] = start + i * step;
    }
    return array;
}