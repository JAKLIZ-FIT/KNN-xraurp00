from collections import deque
from pathlib import Path
import json

def save_last_scores(last_val_loss_scores : deque, save_path : Path):
    with open(save_path/"scores.txt","w") as f: # TODO mark best model
        f.write(f'best epoch= {get_best_score_epoch(last_val_loss_scores)[0]}\n')
        f.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in last_val_loss_scores)) 
        #https://stackoverflow.com/questions/3820312/python-write-a-list-of-tuples-to-a-file

"""
Returns tuple (epoch,val_loss,checkpoint_name)
that has the best val_loss so far
"""
def get_best_score_epoch(last_val_loss_scores):
    return min(last_val_loss_scores, key=lambda item: item[1])
    

#def save_stat_history(stat_history : deque, save_path : Path):
#    with open(save_path/"stat_history.txt","w") as f:
#            f.write('epoch\tcheckpoint\tavgloss\tCAR\tWAR\n')
#            f.write('\n'.join('{}\t{}\t{}\t{}\t{}'.format(x[0],x[1],x[2],x[3],x[4]) for x in stat_history)) 
#            #https://stackoverflow.com/questions/3820312/python-write-a-list-of-tuples-to-a-file

def save_stat_history(stat_history : deque, save_path : Path):
    with open(save_path/"stat_history.txt","w") as f:
            f.write('[epoch, checkpoint, trn_loss_avg, trn_loss_max, trn_loss_min, val_loss_avg, val_loss_max, val_loss_min, CAR, WAR]\n')
            f.write('\n'.join(json.dumps(x) for x in stat_history)) 
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
                if data[0] == 'best':
                    continue
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
    is_first = True
    if not scores_path.exists():
        print('\nNo history found\n')
    else:
        with open(scores_path,"r") as f:
            for line in f.readlines():
                if is_first:
                    is_first=False
                    continue
                #data = line.split()
                #d.append((int(data[0]), data[1], float(data[2]), float(data[3]), float(data[4])))
                d.append(json.loads(line))
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