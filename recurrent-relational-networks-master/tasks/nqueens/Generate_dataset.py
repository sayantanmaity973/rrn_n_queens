import csv

def is_safe(board, row, col):
    # Check if there is a queen in the same column
    for i in range(row):
        if board[i][col] == '1':
            return False
    
    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == '1':
            return False
    
    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, 4)):
        if board[i][j] == '1':
            return False
    
    return True

def solve_4queens_util(board, row):
    # Base case: If all queens are placed, return true
    if row == 4:
        return True

    # Consider this row and try placing queens in all columns
    for col in range(4):
        # Check if the queen can be placed on board[row][col]
        if is_safe(board, row, col):
            # Place this queen in board[row][col]
            board[row][col] = '1'

            # Recur to place rest of the queens
            if solve_4queens_util(board, row + 1):
                return True

            # If placing queen in board[row][col] doesn't lead to a solution,
            # then remove the queen from board[row][col]
            board[row][col] = '0'

    # If the queen cannot be placed in any column in this row,
    # then return false
    return False

def solve_4queens():
    # Initialize an empty board
    board = [['0' for _ in range(4)] for _ in range(4)]

    # Solve the 4-queens problem
    if not solve_4queens_util(board, 0):
        print("No solution exists")
        return None

    # Construct the output string
    output_str = ''.join([''.join(row) for row in board])
    return output_str

def generate_4queens_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        input_str = '0' * 16  # Initial board with all 0's
        output_str = solve_4queens()  # Calculate the output string for the 4-queens problem
        dataset.append((input_str, output_str))
    return dataset

def save_dataset_to_csv(dataset, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input', 'Output'])  # Header row
        writer.writerows(dataset)

# Generate dataset with 100 samples
num_samples = 100
dataset = generate_4queens_dataset(num_samples)

# Save dataset to CSV file
save_dataset_to_csv(dataset, '4queens_dataset_train.csv')
