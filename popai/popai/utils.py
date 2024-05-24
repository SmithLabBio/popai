from collections import Counter

def minor_encoding(arr):
    result = arr.copy()
    for col in range(arr.shape[1]):
        column = arr[:, col]
        column = column[column != -1]
        freq = Counter(column)
        sorted_values = [item[0] for item in freq.most_common()]
        value_to_rank = {value: rank for rank, value in enumerate(sorted_values)}
        for original_value, new_value in value_to_rank.items():
            result[arr[:, col] == original_value, col] = new_value
    return result
