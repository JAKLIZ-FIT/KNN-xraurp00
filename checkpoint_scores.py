from collections import deque
from pathlib import Path

def save_last_scores(last_CAR_scores : deque, save_path : Path):
    with open(save_path+"_scores.txt","w") as f:
            f.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in last_CAR_scores)) 
            #https://stackoverflow.com/questions/3820312/python-write-a-list-of-tuples-to-a-file

def load_last_scores(save_path : Path, maxlen : int):
    """
    Load last maxlen training scores.

    :param save_path: save path for model checkpoints and scores
    :param maxlen: number of last scores to keep

    Returns deque with last scores, empty deque if no previous scores available
    """
    scores_path = save_path + "_scores.txt"
    d = deque(maxlen=maxlen)
    if not (Path(scores_path)).exists():
        print('No previous scores found, starting from epoch 0')
    else:
        with (scores_path,"r") as f:
            for line in f.readlines():
                data = line.split()
                d.append((int(data[0], float(data[1], data[2]))))
    return d