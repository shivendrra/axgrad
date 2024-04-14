#include <stdlib.h>
#include <stdio.h>

typedef struct Array {
    int* data;
    int size;
} Array;

Array* create_array(int* data, int size) {
    Array* array = (Array*)malloc(sizeof(Array));
    if (array == NULL) {
        return NULL; // Memory allocation failed
    }
    array->data = (int*)malloc(size * sizeof(int));
    if (array->data == NULL) {
        free(array); // Free the previously allocated memory
        return NULL; // Memory allocation failed
    }
    for (int i = 0; i < size; ++i) {
        array->data[i] = data[i];
    }
    array->size = size;
    return array;
}

Array* array_add(Array* array1, Array* array2) {
    if (array1->size != array2->size) {
        return NULL; // Arrays must have the same size
    }
    Array* result = create_array(NULL, array1->size);
    if (result == NULL) {
        return NULL; // Memory allocation failed
    }
    for (int i = 0; i < array1->size; ++i) {
        result->data[i] = array1->data[i] + array2->data[i];
    }
    return result;
}

Array* array_sub(Array* array1, Array* array2) {
    if (array1->size != array2->size) {
        return NULL; // Arrays must have the same size
    }
    Array* result = create_array(NULL, array1->size);
    if (result == NULL) {
        return NULL; // Memory allocation failed
    }
    for (int i = 0; i < array1->size; ++i) {
        result->data[i] = array1->data[i] - array2->data[i];
    }
    return result;
}

Array* array_mul(Array* array1, Array* array2) {
    if (array1->size != array2->size) {
        return NULL; // Arrays must have the same size
    }
    Array* result = create_array(NULL, array1->size);
    if (result == NULL) {
        return NULL; // Memory allocation failed
    }
    for (int i = 0; i < array1->size; ++i) {
        result->data[i] = array1->data[i] * array2->data[i];
    }
    return result;
}

Array* array_div(Array* array1, Array* array2) {
    if (array1->size != array2->size) {
        return NULL; // Arrays must have the same size
    }
    Array* result = create_array(NULL, array1->size);
    if (result == NULL) {
        return NULL; // Memory allocation failed
    }
    for (int i = 0; i < array1->size; ++i) {
        if (array2->data[i] == 0) {
            free(result->data); // Free the previously allocated memory
            free(result);
            return NULL; // Division by zero
        }
        result->data[i] = array1->data[i] / array2->data[i];
    }
    return result;
}

void print_array(Array* array) {
    printf("[");
    for (int i = 0; i < array->size; ++i) {
        printf("%d", array->data[i]);
        if (i < array->size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void free_array(Array* array) {
    free(array->data);
    free(array);
}

int main() {
    int data1[] = {1, 2, 3, 4};
    int data2[] = {5, 6, 7, 8};
    Array* array1 = create_array(data1, sizeof(data1) / sizeof(int));
    Array* array2 = create_array(data2, sizeof(data2) / sizeof(int));
    
    Array* result_add = array_add(array1, array2);
    if (result_add != NULL) {
        print_array(result_add);
        free_array(result_add);
    }
    
    Array* result_sub = array_sub(array1, array2);
    if (result_sub != NULL) {
        print_array(result_sub);
        free_array(result_sub);
    }
    
    Array* result_mul = array_mul(array1, array2);
    if (result_mul != NULL) {
        print_array(result_mul);
        free_array(result_mul);
    }
    
    Array* result_div = array_div(array1, array2);
    if (result_div != NULL) {
        print_array(result_div);
        free_array(result_div);
    }
    
    free_array(array1);
    free_array(array2);
    
    return 0;
}