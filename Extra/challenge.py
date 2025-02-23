def sum_of_squares(test_cases, index=0):
    # Base case: If all test cases are processed, return
    if index >= len(test_cases):
        return

    # Get the current test case
    x, numbers = test_cases[index]

    # Filter out negative numbers and calculate squares
    positive_numbers = filter(lambda y: y >= 0, numbers)
    squares = map(lambda y: y * y, positive_numbers)

    # Sum the squares
    total = sum(squares)

    # Print the result for the current test case
    print(total)

    # Process the next test case
    sum_of_squares(test_cases, index + 1)

def main():
    import sys
    input = sys.stdin.read

    # Read all input at once
    data = input().split()

    # Parse the number of test cases
    N = int(data[0])
    test_cases = []
    ptr = 1  # Pointer to track the current position in the input data

    # Process each test case
    for _ in range(N):
        X = int(data[ptr])
        ptr += 1
        numbers = list(map(int, data[ptr:ptr + X]))
        ptr += X
        test_cases.append((X, numbers))

    # Calculate and print results
    sum_of_squares(test_cases)

if __name__ == "__main__":
    main()