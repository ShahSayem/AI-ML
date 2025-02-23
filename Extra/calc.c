#include <stdio.h>

int main() {
    int n1, n2, res;
    printf("Enter value n1 and n2: ");
    scanf("%d %d", &n1, &n2);

    printf("Options:\n1. Add\n2. Subtract\n3. Multiply\n4. Divide\n");

    int option;
    printf("Enter your option: ");
    scanf("%d", &option);

    if (option == 1) {
        res = n1 + n2;
        printf("Addition: %d\n", res);
    } else if (option == 2) {
        res = n1 - n2;
        printf("Subtraction: %d\n", res);
    } else if (option == 3) {
        res = n1 * n2;
        printf("Multiplication: %d\n", res);
    } else if (option == 4) {
        res = n1 / n2;
        printf("Division: %d\n", res);
    } else {
        printf("Invalid option\n");
    }

    return 0;
}
