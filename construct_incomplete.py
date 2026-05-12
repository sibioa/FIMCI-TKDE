import numpy as np
import random
import scipy.io


def splitDigitData(N, V, inCP, seed):
    """
    Function to split digit data based on specified criteria.

    Parameters:
        N (int): The total number of items.
        V (int): The number of groups.B
        inCP (float): The fraction of items to be excluded initially.
        seed (int): The random seed for reproducibility.

    Returns:
        splitInd (numpy.ndarray): An array indicating the split of data.
    """
    # Initialize the split index matrix
    splitInd = np.ones((N, V), dtype=int)

    # Create a list for random permutations of indices
    indCell = [None] * V

    # Calculate the number of elements to delete
    delNum = int(np.floor(N * inCP))

    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)

    # Generate random permutations and initialize the splitInd matrix
    for i in range(V):
        indCell[i] = np.random.permutation(N)
        splitInd[indCell[i][:delNum], i] = 0

    # Counter to track the next index to switch to 0 in each group
    counter = np.array([delNum + 1] * V)

    # Resolve cases where a row in splitInd is all zeros
    while True:
        zerosInd = np.where(np.sum(splitInd, axis=1) == 0)[0]
        if zerosInd.size == 0:
            break
        else:
            i = random.randint(0, V - 1)
            splitInd[zerosInd[0], i] = 1
            if counter[i] < N:  # Check to avoid index error
                splitInd[indCell[i][counter[i]], i] = 0
                counter[i] += 1
    # np.save(r'./data/animal_percentDel_'+str(1-inCP)[:3]+'.npy', splitInd)
    return splitInd


data_name = 'BBC'
missing_rate = 0.9
res = []
for i in range(10):
    res.append(splitDigitData(685,4,missing_rate,i))
data_dict = {
    'folds': res
}
# print(data_dict)
scipy.io.savemat( f'.\\data\\{data_name}_percentDel_{missing_rate}.mat', data_dict)