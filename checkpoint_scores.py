from collections import deque
from pathlib import Path

def save_last_scores(last_CAR_scores : deque, save_path : Path):
    with open(save_path/"scores.txt","w") as f:
            f.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in last_CAR_scores)) 
            #https://stackoverflow.com/questions/3820312/python-write-a-list-of-tuples-to-a-file

def save_stat_history(stat_history : deque, save_path : Path):
    with open(save_path/"stat_history.txt","w") as f:
            f.write('epoch\tcheckpoint\tavgloss\tCAR\tWAR\n')
            f.write('\n'.join('{}\t{}\t{}\t{}\t{}'.format(x[0],x[1],x[2],x[3],x[4]) for x in stat_history)) 
            #https://stackoverflow.com/questions/3820312/python-write-a-list-of-tuples-to-a-file

def load_last_scores(save_path : Path, maxlen : int):
    """
    Load last maxlen training scores.

    :param save_path: save path for model checkpoints and scores
    :param maxlen: number of last scores to keep

    Returns deque with last scores, empty deque if no previous scores available
    """
    scores_path = save_path / "scores.txt"
    d = deque(maxlen=maxlen)
    if not scores_path.exists():
        print('\nNo previous scores found, starting from epoch 0\n')
    else:
        with open(scores_path,"r") as f:
            for line in f.readlines():
                data = line.split()
                d.append((int(data[0]), float(data[1]), data[2]))
    return d


def load_stat_history(save_path : Path):
    """
    Load training history. Includes epoch number, CAR score, WAR score, Loss, checkpoint name.

    :param save_path: save path for model checkpoints and scores

    Returns deque with history, empty deque if no history available
    """
    scores_path = save_path / "stat_history.txt"
    d = deque()
    if not scores_path.exists():
        print('\nNo history found\n')
    else:
        with open(scores_path,"r") as f:
            for line in f.readlines():
                data = line.split()
                d.append((int(data[0]), data[1], float(data[2]), float(data[3]), float(data[4])))
    return d

def load_stat_history_json(save_path : Path):
    scores_path = save_path / "stat_history.json"
    if not scores_path.exists():
        return []
    with open (scores_path, 'r') as f:
        return json.load(f)

def save_stat_history_json(save_path : Path, stat_history):
    scores_path = save_path / "stat_history.json"
    with open (scores_path, 'w') as f:
        f.write(json.dumps(stat_history, indent=2))