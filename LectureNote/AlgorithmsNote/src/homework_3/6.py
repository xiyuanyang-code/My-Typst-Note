from typing import List, Tuple


def preprocessing(
    blocks: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, int]]:
    """
    For each block (a, b, c), generate three oriented blocks:
    Treat each original dimension once as height; the remaining two as (length1, length2),
    ensuring length1 < length2.
    Returns a list of (length1, length2, height).
    """

    oriented_blocks = []

    for a, b, c in blocks:
        dims = [a, b, c]

        # pick each dimension as height once
        for i in range(3):
            h = dims[i]
            l1, l2 = dims[(i + 1) % 3], dims[(i + 2) % 3]

            # ensure l1 < l2
            if l1 > l2:
                l1, l2 = l2, l1

            oriented_blocks.append((l1, l2, h))

    oriented_blocks = list((set(oriented_blocks)))
    blocks_sorted = sorted(oriented_blocks, key=lambda x: -x[0])
    return blocks_sorted


def compute_max_height(blocks: List[Tuple[int, int, int]]):
    blocks_sorted = preprocessing(blocks=blocks)
    len_blocks = len(blocks_sorted)
    x = [0] * len_blocks
    for i in range(len_blocks):
        for j in range(i):
            if (
                blocks_sorted[j][1] > blocks_sorted[i][1]
                and blocks_sorted[j][0] > blocks_sorted[i][0]
            ):
                x[i] = max(x[i], x[j])
        x[i] += blocks_sorted[i][2]

    return max(x)


if __name__ == "__main__":
    n = int(input())
    sequences = []
    for _ in range(n):
        s = input()
        sequences.append(list(map(int, s.split(','))))
    print(compute_max_height(sequences))
