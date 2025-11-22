# AI1804 Assignment 2

This repository contains the coding problems for Assignment 2.

## Directory Structure

- **description/**: Contains the detailed description for each problem in Markdown format. Please read these carefully to understand the requirements and constraints.
- **problem/**: Contains the Python template files for each problem. **You should write your code in these files.**
- **test/**: Contains the unit tests for each problem. You can use these to verify your solutions.

## How to Complete the Assignment

1.  Navigate to the `description/` directory and read the problem descriptions.
2.  Open the corresponding file in the `problem/` directory (e.g., `problem/1_coin_change.py`).
3.  Implement the `Solution` class or the required functions within the file.
    *   **Do not change the file names.**
    *   **Do not change the class names or method signatures.**
4.  Run the tests to verify your implementation.

## How to Run Tests

You can run all tests at once using the following command from the root directory:

```bash
python3 -m unittest discover test
```

To run tests for a specific problem, you can run the corresponding test file directly. For example:

```bash
python3 test/test_1_coin_change.py
```

If your implementation is correct, you should see an output indicating that the tests passed (e.g., `OK`). If there are failures, the output will show which tests failed and why.
