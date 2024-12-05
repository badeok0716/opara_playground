import os
import numpy as np
from examples.io_op import read_pkl

def get_table_content(path2dir, colname, rowname, key='time'):
    filename = f'{rowname}_{colname}.pkl'
    path2file = os.path.join(path2dir, filename)
    try:
        content = read_pkl(path2file)
        if key == 'time':
            tl = content['time_list']
            avg_t, std_t = np.mean(tl), np.std(tl)
            return f'{avg_t:.4f} ({std_t:.4f})'
        elif key == 'mem':
            mem = content['memory']
            return f'{mem:.4f}'
    except Exception as e:
        return '-1'

def get_col_row_names(path2dir):
    files = os.listdir(path2dir)
    cols, rows = set(), set()
    for filename in files:
        assert filename.endswith('.pkl')
        filekey = filename.split('.')[0].split('_')
        assert len(filekey) == 2
        rowname, colname = filekey[0], filekey[1]
        rows.add(rowname)
        cols.add(colname)
    return list(cols), list(rows)

def get_table(path2dir):
    # cols, rows = get_col_row_names(path2dir)
    cols, rows = ['torch', 'torch_compile', 'sequence', 'opara', 'nimble'], ['deepfm', 'googlenet', 'nasnet']
    table = [[''] + cols]
    for row in rows:
        row_content = [row]
        for col in cols:
            row_content.append(get_table_content(path2dir, col, row))
        table.append(row_content)
    return table

# print table in csv
def print_table(table):
    for row in table:
        print(','.join(row))

if __name__ == '__main__':
    rootdir = '/home/deokjae/Opara/examples/results_2.0.0/'

    for expkey in [f'{nodename}_{nbatch}_1000_5000' for nodename in ['alpaca', 'david', 'donut', 'vanilla', 'zinger'] for nbatch in [16]]:
        table = get_table(rootdir + expkey)
        expstr = f'infer time (ms) for {expkey}'
        print(expstr)
        print_table(table)
        print()
