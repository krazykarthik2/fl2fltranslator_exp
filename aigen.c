#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>

/* ====== Typedefs ====== */
typedef unsigned long ulong_t;
typedef struct Node Node;

/* ====== Global Variables ====== */
extern int external_var;
int external_var = 42;

static int static_global = 100;
volatile int volatile_global = 5;
const double const_global = 3.14159;

/* ====== Enum ====== */
enum Color {
    RED = 1,
    GREEN,
    BLUE = 10
};

/* ====== Struct with bitfields ====== */
struct Flags {
    unsigned int a : 1;
    unsigned int b : 2;
    unsigned int c : 5;
};

/* ====== Union ====== */
union Data {
    int i;
    float f;
    double d;
};

/* ====== Flexible Array Member ====== */
struct Packet {
    size_t length;
    char data[];
};

/* ====== Self-referential Struct ====== */
struct Node {
    int value;
    Node *next;
};

/* ====== Alignment ====== */
_Alignas(16) struct Aligned {
    char c;
    int i;
};

/* ====== Static Assert ====== */
_Static_assert(sizeof(int) >= 2, "int too small");

/* ====== Inline Function ====== */
static inline int add(int a, int b) {
    return a + b;
}

/* ====== Function Pointer ====== */
typedef int (*op_func)(int, int);

/* ====== Variadic Function ====== */
int sum_variadic(int count, ...) {
    va_list args;
    va_start(args, count);
    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);
    }
    va_end(args);
    return total;
}

/* ====== Recursive Function ====== */
int factorial(int n) {
    if (n <= 1)
        return 1;
    return n * factorial(n - 1);
}

/* ====== _Generic Example ====== */
#define type_name(x) _Generic((x), \
    int: "int", \
    float: "float", \
    double: "double", \
    default: "other")

/* ====== Noreturn Function ====== */
_Noreturn void fatal(const char *msg) {
    fprintf(stderr, "Fatal: %s\n", msg);
    exit(EXIT_FAILURE);
}

/* ====== Complex Function ====== */
int complex_function(int argc, char *argv[]) {
    auto int x = 10;
    register int y = 20;
    int result = 0;

    unsigned char uc = 255;
    signed char sc = -1;
    short s = -10;
    unsigned short us = 10;
    long l = 1000L;
    unsigned long ul = 1000UL;
    long long ll = -100000LL;
    unsigned long long ull = 100000ULL;

    float f = 1.23f;
    double d = 4.56;
    long double ld = 7.89L;
    _Bool flag = 1;

    int arr[3][3] = {
        [0] = {1,2,3},
        [1] = {4,5,6},
        [2] = {7,8,9}
    };

    struct Flags flags = { .a = 1, .b = 2, .c = 15 };
    union Data data;
    data.i = 42;

    op_func op = add;

    /* Loops */
    for (int i = 0; i < 3; i++) {
        int j = 0;
        while (j < 3) {
            result += arr[i][j];
            j++;
        }
    }

    do {
        result--;
    } while (result > 50);

    /* Switch */
    switch (flags.a) {
        case 0:
            result += 1;
            break;
        case 1:
            result += 2;
            break;
        default:
            result += 3;
    }

    /* Ternary */
    result += (flag ? 10 : -10);

    /* Comma operator */
    int temp = (x++, y++, x + y);

    /* Goto */
    if (temp > 0)
        goto label;

    label:
        result += temp;

    /* Pointer operations */
    int value = 5;
    int *ptr = &value;
    int **pptr = &ptr;
    result += **pptr;

    /* Compound literal */
    int *cl = (int[]){1,2,3,4};
    result += cl[2];

    /* sizeof and alignof */
    result += sizeof(ld);
    result += _Alignof(struct Aligned);

    /* Cast */
    result += (int)f;

    /* Variadic */
    result += sum_variadic(3, 1, 2, 3);

    /* Recursive */
    result += factorial(5);

    printf("Type of result: %s\n", type_name(result));

    return result;
}

/* ====== Main ====== */
int main(int argc, char *argv[]) {
    int r = complex_function(argc, argv);
    printf("Final result: %d\n", r);

    if (argc > 1000) {
        fatal("Too many arguments");
    }

    return 0;
}