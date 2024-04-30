import matplotlib.pyplot as plt
import pandas as pd
from checkpoint_scores import *
import sys
import argparse


#def parse_args():
#    parser = argparse.ArgumentParser('Plot metrics from training history.')
#    parser.add_argument(
#        '-s', '--stat-path',
#        help='Path to file with metrics history.',
#        type=Path,
#        required=True
#    )
#    return parser.parse_args()

def main():
    #args = parse_args()

    # TODO generate automatically
    stat_names = ['epoch', 'checkpoint', 'trn_loss_avg', 'trn_loss_max', 'trn_loss_min', 
                  'val_loss_avg', 'val_loss_max', 'val_loss_min', 'CAR', 'WAR']

    #stat_history, _ = load_stat_history(args.stat_path)
    stat_history, _ = load_stat_history(Path("models/small_stage1_augmented_trn"))
    print(stat_history[0])
    stats_df = pd.DataFrame(list(stat_history), columns=list(stat_names))

    print(stats_df)
    x = range(1,len(stats_df.val_loss_avg)+1)
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.scatter([3],[stats_df.val_loss_avg.min()],color='C0')
    ax1.plot(x, stats_df.val_loss_avg, color='C0', label='val_avg')
    ax1.plot(x, stats_df.val_loss_max, color='C0', label='val_max')
    ax1.plot(x, stats_df.val_loss_min, color='C0', label='val_min')
    #ax1.plot(x, stats_df.trn_loss_avg, 'o--', color='C1', label='trn_avg')
    ax1.scatter([11],[stats_df.trn_loss_avg.min()],color='C1')
    ax1.plot(x, stats_df.trn_loss_avg, color='C1', label='trn_avg')
    ax1.plot(x, stats_df.trn_loss_max, color='C1', label='trn_max')
    ax1.plot(x, stats_df.trn_loss_min, color='C1', label='trn_min')
    ax1.fill_between(x, stats_df.val_loss_avg, stats_df.val_loss_max, color='C0', alpha=0.25)
    ax1.fill_between(x, stats_df.val_loss_avg, stats_df.val_loss_min, color='C0', alpha=0.25)
    ax1.fill_between(x, stats_df.trn_loss_avg, stats_df.trn_loss_max, color='C1', alpha=0.25)
    ax1.fill_between(x, stats_df.trn_loss_avg, stats_df.trn_loss_min, color='C1', alpha=0.25)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.scatter([3],[stats_df.CAR.max()],color='C0',label='best CAR')
    ax2.scatter([3],[stats_df.WAR.max()],color='C1',label='best WAR')
    ax2.plot(x, stats_df.CAR, color='C0', label='CAR')
    ax2.plot(x, stats_df.WAR, color='C1', label='WAR')
    plt.legend()
    fig.show()
    
    #data = pd.read_csv(args.stat_path, sep=" ")
    #print(data)
    
    # TODO plot metrics

    return 0


if __name__ == '__main__':
    sys.exit(main())
