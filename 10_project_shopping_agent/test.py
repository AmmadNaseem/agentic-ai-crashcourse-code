def queen_fitness(board):

    attacks = 0
    n = len(board)

    for i in range(n):
        for j in range(i + 1, n):

            same_row = board[i] == board[j]

            same_diagonal = (
                abs(board[i] - board[j])
                == abs(i - j)
            )

            if same_row or same_diagonal:
                attacks += 1

    total_pairs = n * (n - 1) // 2

    return total_pairs - attacks


board = [1, 5, 8, 6, 3, 7, 2, 4]

print("Chromosome:", board)
print("Fitness:", queen_fitness(board))
