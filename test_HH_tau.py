import numpy as np

rng = np.random.default_rng(24601)

def generate_data(n, d, tau):
    '''
        Function:
            generate 2 vectors of length n living in hamming space
            shift distance at most d with tau shifts
        
        Input:
            n: vector length
            d: shift distance upper bound
            tau: no. of shifts

        Output:
            v1, v_2: arrays, two vectors
    '''

    v1 = rng.integers(2, size = n)
    v2 = v1.copy()
    idx = rng.choice(n, d, replace=False)
    v2[idx] = 1 - v2[idx]
    v2 = np.roll(v2, tau)
    return v1, v2

def sh(v1, v2):
    f = lambda tau: H(v1, np.roll(v2, tau))
    ham_distances = [f(tau) for tau in range( len(v1))]

    min_pos = np.argmin(ham_distances)
    shift_distance = ham_distances[min_pos]

    tau = min( min_pos, len(v1) - min_pos)

    return shift_distance, tau

def H(v1, v2):
    return np.linalg.norm((v1-v2), ord = 1)

def count_frequency(arr, k):
    '''
        Function:
            Count how many times each subarray appears. Store the positions in an array
        
        Input:
            arr: the array which we want to count its subarrays
            k: length of subarray
        
        Output:
            subarrays: dictionary that marks the position where each subarray occurs
                keys: tuple of the corresponding subarray
                values: an array indicating where the subarray appears
    '''

    subarrays = {}
    for i in range(len(arr) - k + 1):
        subarray = tuple(arr[i:i+k])
        if subarray in subarrays:
            subarrays[subarray].append(i)
        else:
            subarrays[subarray] = [i]
    return subarrays
