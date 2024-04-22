from dataclasses import dataclass
from collections import deque
from pathlib import Path
import json
import argparse
from checkpoint_scores import *


@dataclass
class TrainConfig:
    model : Path

    training_dataset : Path
    validation_dataset : Path
    
    training_labels : Path
    validation_labels : Path

    epochs : int 
    save_path : Path
    batch_size : int
    use_gpu: bool
    num_checkpoints : int
    early_stop_threshold: float
    use_config : bool
    config_path : Path
    start_epoch: int = 0
    last_CAR_scores: deque = None
    stat_history: deque = None
    #stat_history_json = None
    


def load_config(args):
    use_config = args.use_config
    pre_config = {}
    if use_config:
        if not args.config_path.exists():
            raise ValueError(f'Path {args.config_path} does not exist!')
        with open(args.config_path,"r") as cf:
            pre_config = json.loads(cf.read())
    else:
        pre_config['model'] = str(args.model)
        pre_config['training_dataset'] = str(args.training_dataset)
        pre_config['validation_dataset'] = str(args.validation_dataset)
        pre_config['training_labels'] = str(args.training_labels)
        pre_config['validation_labels'] = str(args.validation_labels)
        pre_config['epochs'] = args.epochs
        pre_config['save_path'] = str(args.save_path)
        pre_config['batch_size'] = args.batch_size
        pre_config['use_gpu'] = args.use_gpu
        pre_config['num_checkpoints'] = args.num_checkpoints
        pre_config['early_stop'] = args.early_stop
        pre_config['config_path'] = str(args.config_path)
        pre_config['start_epoch'] = 0

        save_config(pre_config)
        # TODO save with current epoch?
        # TODO save even when restarting? it could reset the start_epoch variable
        # right now it will look for file with last CAR scores to see what model to load, so saving epoch not necessary 
    
    print("===============================")
    print("======= Training config =======")
    for x,y in pre_config.items():
        print(f"{x}\t= {y}")
    print("===============================")

    # load last scores (also get last model checkpoint)
    last_CAR_scores = load_last_scores(Path(pre_config['save_path']),pre_config['num_checkpoints'])
    # TODO select last model
    start_epoch = 0
    if len(last_CAR_scores) > 0:
        last_score = last_CAR_scores[-1]
        pre_config['model'] = pre_config['save_path'] +"/"+ last_score[2]
        start_epoch = last_score[0]
        print(f"\nfound checkpoint from epoch {start_epoch}")
        print(f"path: {pre_config['model']}")
        print(f"score: {last_score[1]}\n")

    #stat_history_json = load_stat_history_json(Path(pre_config['save_path']))
    stat_history = load_stat_history(Path(pre_config['save_path']))

    return TrainConfig(
        model=Path(pre_config['model']),
        training_dataset=Path(pre_config['training_dataset']),
        validation_dataset=Path(pre_config['validation_dataset']) if pre_config['validation_dataset'] != 'None' else None,
        training_labels=Path(pre_config['training_labels']),
        validation_labels=Path(pre_config['validation_labels']),
        epochs=pre_config['epochs'],
        save_path=Path(pre_config['save_path']),
        batch_size=pre_config['batch_size'],
        use_gpu=pre_config['use_gpu'],
        num_checkpoints=pre_config['num_checkpoints'],
        early_stop_threshold=pre_config['early_stop'],
        use_config=use_config,
        config_path=Path(pre_config['config_path']),
        start_epoch=start_epoch,
        last_CAR_scores=last_CAR_scores,
        stat_history=stat_history
    )


def save_config(config):
    #with open(,"w") as cf: # TODO select this if config is actual config class instance
    config_path = config['save_path']/"config.json" if type(config) == TrainConfig else config['save_path']+"/config.json"
    if not config_path.exists(): # save checkpoint
                os.makedirs(config_path)
    with open(config_path,"w") as cf:
        cf.write(json.dumps(config,indent=2))
    