
def get_new_batch_size(length, batch_size):
    optimal_batch_size = 0

    for i in range(batch_size, 2, -1):
        if length % i >= 3:
            optimal_batch_size = max(optimal_batch_size, i)

    return optimal_batch_size
