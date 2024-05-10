#!/usr/bin/env python3

import matplotlib.pyplot as plt
import json
import numpy
import sys

def mkgraph():
    f = open('bucket20_metrics.json', 'r')
    j = json.load(f)
    f.close()

    x = numpy.arange(1, len(j['conf_product']['vals']) + 1)
    y = [j['conf_product']['vals'], j['conf_mean']['vals'], j['conf_cn']['vals']]
    plt.plot(x, y[0], label='Product')
    plt.plot(x, y[1], label='Mean')
    plt.plot(x, y[2], label='CN')
    plt.ylabel(ylabel='Cumulative CER [%]')
    plt.xlabel(xlabel='Number of the most confident data samples')
    plt.title(label='Confidence types (20 buckets).')
    plt.legend()
    #plt.show()
    plt.savefig('metrics.png')
    return 0

if __name__ == '__main__':
    sys.exit(mkgraph())
