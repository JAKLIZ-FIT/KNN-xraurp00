#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import random
from evaluate import load
from pero_ocr.decoding.confusion_networks import add_hypothese
import json
import argparse
import sys

class MetricsEvaluator:
    """
    Evaluates confidence metrics.
    """
    def __init__(self, file: Path, separator: str) -> None:
        """
        Initializes evaluator.
        :param file (pathlib.Path): path to file with dataset to evaluate.
        :param separator (str): dataset field separator
        """
        self.augmentation_exts = [
            '_gaussian_square.jpg',
            '_resize.jpg',
            '_skewed.jpg',
            '_mask.jpg',
            '_noise.jpg',
            '_rotate.jpg',
            '_mixup.jpg',
            '_cutmix.jpg'
        ]
        self.dataset = self.load_stat_file(file=file, separator=separator)
    
    def check_augmentation(self, file_name: str) -> bool:
        """
        Checks if file_name belongs to augmented file.
        :param file_name (str): file_name to check.
        :return (bool): True if label marks augmented file, False otherwise.
        """
        for ext in self.augmentation_exts:
            if len(ext) >= len(file_name):
                continue
            if file_name[-len(ext):] == ext:
                return True
        return False

    def load_stat_file(self, file: Path, separator: str) -> dict:
        """
        Loads file with statistics and transcripted validation dataset.
        :param file (pathlib.Path): path to file
        :param separator (str): dataset field separator
        :return (dict): dataset loaded as dictionary, indexed by file names
        """
        cer = load('cer')
        d = {}
        df = pd.read_csv(file, index_col=0, sep=separator)
        for i, r in df.iterrows():
            v = {}
            v['references'] = r.references if pd.notna(r.references) else ''
            v['predictions'] = r.predictions if pd.notna(r.predictions) else ''
            if pd.notna(r.Conf_product):
                v['conf_product'] = r.Conf_product
            else:
                v['conf_product'] = 0.0
            v['conf_mean'] = r.Conf_mean if pd.notna(r.Conf_mean) else 0.0
            v['conf_cn'] = 0.0
            v['augmented'] = self.check_augmentation(
                file_name=r.filenames
            )
            v['cer'] = cer.compute(
                predictions=[v['predictions']],
                references=[v['references']]
            )
            d[r.filenames] = v
        return d
    
    def gen_confusion_net_confidence(self, predictions: list[str]) -> float:
        """
        Generates confidence based on confusion network for given predictions.
        :param predictions (list[str]): predictions to generate confusion
            network for
        :return (float): confidence score (1.0 / confusion score)
        """
        cn = []
        for prediction in predictions:
            cn = add_hypothese(cn, prediction, 1)
        confusion = np.prod(list(map(len, cn)))
        return 1.0 / confusion if confusion else 1.0
    
    def get_augmented_file_versions(self, file_name: str) -> list[str]:
        """
        Returns names of augmented versions of the data file.
        :param file_name (str): name of non-augmented file
        :returns (list[str]): list of names of augmented data files
        """
        ext = '.jpg'
        alt_file_names = []

        if file_name[-len(ext):] != ext:
            raise ValueError(f'File name does not end with {ext} extension!')

        file_name_no_ext = file_name[:-len(ext)]
        for aext in self.augmentation_exts:
            alt_file_names.append(file_name_no_ext + aext)
        
        return alt_file_names

    def compute_cn_conf(self) -> None:
        """
        Calculates confusion network confidences for whole dataset.
        """
        for file_name, data in self.dataset.items():
            if data['augmented']:
                continue

            augmented_data_files = self.get_augmented_file_versions(
                file_name=file_name
            )
            predictions = [data['predictions']]
            for f in augmented_data_files:
                if f in self.dataset:
                    predictions.append(self.dataset[f]['predictions'])
            
            data['conf_cn'] = self.gen_confusion_net_confidence(
                predictions=predictions
            )
    
    def eval_metric(
        self,
        confidence_type: str
    ) -> (list[float], list[float], float):
        """
        Evaluates how well given confidence type corespondes to the CER.
        CER corespondence is computed as
            sum(CER of results) / number of results.
        -> Smaller area under curve means better result! <-
        :param confidence_type (str): name of confidence field to evaluate
        :returns (list[float], float): tuple(
            confidences,
            CER corespondence,
            area under the curve for given metric
        )
        """
        # get confidence and cer
        conf_cer = []
        for file_name, data in self.dataset.items():
            if data['augmented']:
                continue
            conf_cer.append((data[confidence_type], data['cer']))
        
        # sort from highest confidence to lowest
        conf_cer.sort(key=lambda x: x[0], reverse=True)

        # get only the cer values
        cer_vals = [x[1] for x in conf_cer]

        # calcualte cer cumsum
        cer_vals = np.cumsum(cer_vals).tolist()

        # divide by number of samples for each position
        for i in range(len(cer_vals)):
            cer_vals[i] = cer_vals[i] / (i+1)

        # calculate area under curve
        auc = np.trapz(cer_vals)

        return [x[0] for x in conf_cer], cer_vals, auc
    
    def get_buckets(
        self,
        n_buckets: int = 10
    ) -> (list[list[str]], int):
        """
        Creates buckets based on lenght of the predicted labels.
        :param n_buckets (int): number of buckets to create
        :returns (list[list[str]], int): (
            list of buckets of data ids/keys,
            size of largest bucket
        )
        """
        # get filenames of non-augmented data
        filenames = []
        for file_name, data in self.dataset.items():
            if data['augmented']:
                continue
            filenames.append(file_name)
        
        # sort by transcription length
        filenames.sort(key=lambda x: len(self.dataset[x]['predictions']))

        # create buckets
        buckets = []
        step = int(len(filenames) / n_buckets)
        partition_start = 0
        partition_end = step
        for i in range(n_buckets - 1):
            bucket = filenames[partition_start:partition_end]
            buckets.append(bucket)
            partition_start = partition_end
            partition_end += step
        # align last partition
        partition_end = len(filenames)
        buckets.append(filenames[partition_start:partition_end])

        # get longest bucket
        max_size = max(map(len, buckets))

        return buckets, max_size
    
    def sort_bucket_by_field(
        self,
        buckets: list[list[str]],
        field_name: str
    ) -> None:
        """
        Sorts buckets by given field.
        :param buckets (list[list[str]]): list of buckets of data ids/keys
        :param field_name (str): name of field to sort buckets by
        :returns: None (buckets are modified in-place)
        """
        for bucket in buckets:
            bucket.sort(
                key=lambda x: self.dataset[x][field_name],
                reverse=True
            )

    def eval_metric_buckets(
        self,
        confidence_type: str,
        n_buckets: int = 10
    ) -> (list[float], float):
        """
        Evaluates how well given confidence type corespondes to the CER.
        Evaluation is done for data separated to buckets based on transcription
        length. Data are taken from bucket according to their confidence.
        CER corespondence is computed as
            sum(CER of results) / number of results.
        -> Smaller area under curve means better result! <-
        :param confidence_type (str): name of confidence field to evaluate
        :param n_buckets (int): number of buckets to separate dataset to
        :returns (list[float], float): tuple(
            CER values,
            area under the curve for given metric
        )
        """
        # get buckets and largest bucket size
        buckets, n_steps = self.get_buckets(n_buckets=n_buckets)

        # sort each bucket by confidence
        self.sort_bucket_by_field(buckets=buckets, field_name=confidence_type)

        cer_vals = []
        # calculate metric for data taken from buckets
        for i in range(n_steps):
            tmp = []
            for bucket in buckets:
                tmp.extend(bucket[:i + 1])
            tmp_cer = [self.dataset[fn]['cer'] for fn in tmp]
            cer_vals.append(np.sum(tmp_cer) / ((i + 1) * n_buckets))
        
        # calculate area under curve
        auc = np.trapz(cer_vals)

        return cer_vals, auc

    def bucket_select(
        self,
        confidence_type: str,
        n_buckets: int = 10,
        percentages: list[int] = [100],
        output_file_prefix: str = None
    ) -> None:
        """
        Selects data from buckets to be used in training.
        :param confidence_type (str): confidence type to use for selection
        :param n_buckets (int): number of buckets to use
        :param percentages (list[int]): percentages of data to select
        :param output_file_prefix (str): output file prefix
        :returns: None (generates output to files)
        """
        # get buckets and largest bucket size
        buckets, bucket_size = self.get_buckets(n_buckets=n_buckets)

        # sort each bucket by confidence
        self.sort_bucket_by_field(buckets=buckets, field_name=confidence_type)

        # set prefix for output file
        if not output_file_prefix:
            output_file_prefix = confidence_type + '_bucket_'
        
        # generate output files
        for percentage in percentages:
            output_file = open(output_file_prefix + str(percentage), 'w')

            # get output data
            output = []
            for bucket in buckets:
                output.extend(bucket[:int(bucket_size*percentage/100)])
            
            # get augmented versions of augmented data
            for fn in output:
                aug = self.get_augmented_file_versions(fn)
                for a in aug:
                    if a in self.dataset:
                        output.append(a)
            
            random.shuffle(output)

            # write output to file
            for key in output:
                output_file.write(
                    f'{key} 0 {self.dataset[key]["predictions"]}\n'
                )

            output_file.close()

    def save_bucket_metrics_to_file(
        self,
        file: Path,
        confidence_types: list[str],
        n_buckets: int
    ):
        """
        Saves metrics to json file after evaluation.
        :param file (pathlib.Path): path to file
        :param confidence_types (list): list of confidence types to evaluate
        :param n_buckets (int): number of buckets to use
        :return (dict): results (same as saved to file)
        """
        results = {}
        file = open(file, 'w')

        for ct in confidence_types:
            vals, area = self.eval_metric_buckets(
                confidence_type=ct,
                n_buckets=n_buckets
            )
            results[ct] = {
                'conf': [],
                'vals': vals,
                'auc': area
            }
        
        json.dump(results, file, indent=4)
        file.close()

        return results

    def save_metrics_to_file(
        self,
        file: Path,
        confidence_types: list[str]
    ) -> dict:
        """
        Saves metrics to json file after evaluation.
        :param file (pathlib.Path): path to file
        :param confidence_types (list): list of confidence types to evaluate
        :return (dict): results (same as saved to file)
        """
        results = {}
        file = open(file, 'w')

        for ct in confidence_types:
            conf, vals, area = self.eval_metric(confidence_type=ct)
            results[ct] = {
                'conf': conf,
                'vals': vals,
                'auc': area
            }
        
        json.dump(results, file, indent=4)
        file.close()

        return results
    
    def save_intermediate_result(self, output_file: Path):
        """
        Saves loaded dataset in current state to json file, so it can be loaded
        later to restore computation.
        :param output_file
        """
        with open(output_file, 'w') as output:
            json.dump(self.dataset, output, indent=4)

def parse_args():
    parser = argparse.ArgumentParser('Evaluates confidence metrics.')
    parser.add_argument(
        '-d', '--dataset',
        help='Dataset to evaluate metrics on use.',
        type=Path
    )
    parser.add_argument(
        '-o', '--output-file',
        help='Output file to write results to.',
        type=Path
    )
    parser.add_argument(
        '-m', '--metrics-to-evaluate',
        help='Confidence metrics names to evaluate (names of their fields in '
             'dataset).',
        nargs='+',
        default=[]
    )
    parser.add_argument(
        '-b', '--number-of-buckets',
        help='Number of buckets to split data to.',
        type=int,
        default=0
    )
    parser.add_argument(
        '-s', '--selected-metric',
        help='Metric selected to use for spliting data based on otput '
             'percentage.',
        default=None
    )
    parser.add_argument(
        '-x', '--output-file-prefix',
        help='Prefix of output files when saving buckets.',
        default=None
    )
    parser.add_argument(
        '-p', '--bucket-percentages',
        help='Percentage (as whole numbers) of data selected data in output '
             'file. Can be multiple values, to create multiple files.',
        nargs='+',
        type=int,
        default=[100]
    )
    parser.add_argument(
        '--separator',
        help='Dataset field separator. (default=\',\')',
        default=','
    )
    parser.add_argument(
        '-i', '--save-intermediate-result',
        help='Saves dataset to json file after confision networks are '
             'computed.',
        type=Path,
        default=None
    )
    return parser.parse_args()

def main():
    args = parse_args()
    me = MetricsEvaluator(args.dataset, separator=args.separator)
    me.compute_cn_conf()
    if args.save_intermediate_result:
        me.save_intermediate_result(args.save_intermediate_result)
    if not args.selected_metric:
        if not args.number_of_buckets:
            metrics = me.save_metrics_to_file(
                args.output_file,
                args.metrics_to_evaluate
            )
        else:
            metrics = me.save_bucket_metrics_to_file(
                args.output_file,
                args.metrics_to_evaluate,
                n_buckets=args.number_of_buckets
            )
        for metric, values in metrics.items():
            print(f'Metric name: {metric}, metric auc: {values["auc"]}.')
    else:
        me.bucket_select(
            confidence_type=args.selected_metric,
            n_buckets=args.number_of_buckets,
            percentages=args.bucket_percentages,
            output_file_prefix=args.output_file_prefix
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
