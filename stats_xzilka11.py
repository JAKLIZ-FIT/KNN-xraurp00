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
    #stat_history, _ = load_stat_history(Path("models/small_stage1_augmented_trn"))
    stat_history, _ = load_stat_history(Path("models/base_stage1_augmented_trn"))
    print(stat_history[0])
    stats_df = pd.DataFrame(list(stat_history), columns=list(stat_names))

    print(stats_df)
    x = range(1,len(stats_df.val_loss_avg)+1)
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    stats_idxmin = stats_df.idxmin()
    val_loss_avg_idxmin = stats_idxmin['val_loss_avg']+1
    trn_loss_avg_idxmin = stats_idxmin['trn_loss_avg']+1
    stats_idxmax = stats_df.idxmax()
    car_idxmax = stats_idxmax['CAR']+1
    war_idxmax = stats_idxmax['WAR']+1
    
    stats_df['CER'] = 1-stats_df['CAR']
    stats_df['WER'] = 1-stats_df['WAR']
    
    ax1.scatter([val_loss_avg_idxmin],[stats_df.val_loss_avg.min()],color='C0',label='trn_best')
    ax1.plot(x, stats_df.val_loss_avg, color='C0', label='val_avg')
    ax1.plot(x, stats_df.val_loss_max, color='C0', label='val_max')
    ax1.plot(x, stats_df.val_loss_min, color='C0', label='val_min')
    #ax1.plot(x, stats_df.trn_loss_avg, 'o--', color='C1', label='trn_avg')
    ax1.scatter([trn_loss_avg_idxmin],[stats_df.trn_loss_avg.min()],color='C1',label='trn_best')
    ax1.plot(x, stats_df.trn_loss_avg, color='C1', label='trn_avg')
    ax1.plot(x, stats_df.trn_loss_max, color='C1', label='trn_max')
    ax1.plot(x, stats_df.trn_loss_min, color='C1', label='trn_min')
    ax1.fill_between(x, stats_df.val_loss_avg, stats_df.val_loss_max, color='C0', alpha=0.25)
    ax1.fill_between(x, stats_df.val_loss_avg, stats_df.val_loss_min, color='C0', alpha=0.25)
    ax1.fill_between(x, stats_df.trn_loss_avg, stats_df.trn_loss_max, color='C1', alpha=0.25)
    ax1.fill_between(x, stats_df.trn_loss_avg, stats_df.trn_loss_min, color='C1', alpha=0.25)
    
    # plot CAR,WAR
    #ax2.set_title('Accuracy')
    #ax2.set_xlabel('Epoch')
    #ax2.set_ylabel('Accuracy')
    #ax2.scatter([car_idxmax],[stats_df.CAR.max()],color='C0',label='best CAR')
    #ax2.scatter([war_idxmax],[stats_df.WAR.max()],color='C1',label='best WAR')
    #ax2.plot(x, stats_df.CAR, color='C0', label='CAR')
    #ax2.plot(x, stats_df.WAR, color='C1', label='WAR')
    
    #plot CER,WER
    ax2.set_title('Error rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error rate')
    ax2.scatter([car_idxmax],[stats_df.CER.min()],color='C0',label='best CER')
    ax2.scatter([war_idxmax],[stats_df.WER.min()],color='C1',label='best WER')
    ax2.plot(x, stats_df.CER, color='C0', label='CER')
    ax2.plot(x, stats_df.WER, color='C1', label='WER')
    
    plt.legend()
    fig.show()
    
    #data = pd.read_csv(args.stat_path, sep=" ")
    #print(data)
    
    # TODO plot metrics

    return 0


if __name__ == '__main__':
    sys.exit(main())
