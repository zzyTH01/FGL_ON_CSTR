import numpy as np

def group_seizure(X, y, onset_indices):
    Xg = []
    yg = []
    print ('onset_indices', onset_indices)

    for i in range(len(onset_indices)-1):
        Xg.append(
            np.concatenate(X[onset_indices[i]:onset_indices[i+1]], axis=0)
        )
        yg.append(
            np.array(y[onset_indices[i]:onset_indices[i+1]])
        )
    return Xg, yg