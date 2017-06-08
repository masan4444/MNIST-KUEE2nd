#include <string.h>
#include "nn.h"
#include "nnlib.h"

int main() {
    float A[6] = {1, 2, 3, 4, 5, 6};
    float b[2] = {1, 2};
    float x[3] = {2, 4, 2};
    float y[2];
    fc(2, 3, x, A, b, y);
    print(2, 1, y);
    return 0;
}
