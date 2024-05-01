#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from evaluate import load
from pero_ocr.decoding.confusion_networks import add_hypothese
import json
import argparse
import sys

class MetricsEvaluator:
    """
    Evaluates confidence metrics.
    """
    def __init__(self, file: Path) -> None:
        """
        Initializes evaluator.
        :param file (pathlib.Path): path to file with dataset to evaluate.
        """
        self.augmentation_exts = [
            '_gaussian_square.jpg',
            '_resize.jpg',
            '_skewed.jpg',
            '_mask.jpg',
            '_noise.jpg'
        ]
        self.dataset = self.load_stat_file(file=file)
    
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

    def load_stat_file(self, file: Path) -> dict:
        """
        Loads file with statistics and transcripted validation dataset.
        :param file (pathlib.Path): path to file
        :return (dict): dataset loaded as dictionary, indexed by file names
        """
        cer = load('cer')
        d = {}
        df = pd.read_csv(file, index_col=0)
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
        return 1.0 / confusion
    
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
    return parser.parse_args()

def main():
    args = parse_args()
    me = MetricsEvaluator(args.dataset)
    me.compute_cn_conf()
    metrics = me.save_metrics_to_file(args.output_file, args.metrics_to_evaluate)
    for metric, values in metrics.items():
        print(f'Metric name: {metric}, metric auc: {values["auc"]}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
