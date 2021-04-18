# -*- coding:utf-8 -*-
import os
import json
import random
import shutil
import subprocess
import tarfile

from tqdm import tqdm

from .command import run_command
from .log import logger


def combine_file(in_file_big, in_file_small, out_file, small_times=1, remove=False):
    """Combine a small file to a big file and save the combined data in `in_file_big`.comb.

    Args:
        in_file_big: Big file path.
        in_file_small: Small file path.
        out_file:
        small_times: Number of small data duplications.

    Returns:
        Output file name.
    """
    logger.info(f'Combining files {in_file_big} and {in_file_small}.')
    assert os.path.exists(in_file_big)
    assert os.path.exists(in_file_small)
    shutil.copy(in_file_big, out_file)
    for i in range(small_times):
        run_command(f'cat {in_file_small} >> {out_file}')
    if remove:
        os.remove(in_file_big)
        os.remove(in_file_small)
    return out_file


def enumerate_file(file_path):
    """Enumerate file.

    Args:
        file_path: File path.

    Yields:
        line_no: Line number in the file.
        line: File line.
    """
    length = file_len(file_path)
    with open(file_path, 'r', encoding='utf-8') as in_file:
        for line_no, line in enumerate(tqdm(in_file, ncols=120, desc=f'Reading {file_path}', total=length)):
            line = line.rstrip('\n')
            if line == '':
                continue
            yield line_no, line


def enumerate_file_list(file_list):
    """Enumerate list of file.

    Args:
        file_list: List of file path.

    Yields:
        line_no: Line number in the file.
        line: File line.
    """
    for file_path in file_list:
        assert os.path.exists(file_path), f'File {file_path} does not exist.'
        for line_no, line in enumerate_file(file_path):
            yield line_no, line


def file_len(file_name):
    """Get the number of lines in file.

    Args:
        file_name: File name.

    Returns:
        Number of lines.
    """
    p = subprocess.Popen(['wc', '-l', file_name],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def load_json(json_file):
    """Load json file."""
    logger.info(f'Loading json from {json_file}.')
    assert os.path.exists(json_file), f'Json file {json_file} does not exist.'
    with open(json_file, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    return data_json


def save_json(data, json_file):
    """Save json file."""
    logger.info(f'Saving json to {json_file}.')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def shuffle(in_file, out_file=None):
    """Shuffle file by selecting an appropriate method according to the size of input file.

    If data size > 5GB, the method will call `shuffle_big_data` function to shuffle file. Otherwise,
    the method calls `shuffle_small_data` function.

    Args:
        in_file: Input file path.
        out_file: Output file path. The default is `in_file`.shuf.

    Returns:
        None.
    """
    logger.info(f'Shuffle file {in_file}.')
    if out_file is None:
        out_file = in_file + '.shuf'

    size = os.path.getsize(in_file)
    if size > (1024 ** 3) * 5:
        shuffle_big_data(in_file, out_file)
    else:
        shuffle_small_data(in_file, out_file)


def shuffle_by_command(in_file, out_file, temp_dir):
    """Shuffle file by another method.

    1. Assign a random value (0-1) to each line in `in_file`;
    2. Sort the lines according their assigned values;
    3. Output each line in order to `out_file`.

    Args:
        in_file: Input file path.
        out_file: Output file path.
        temp_dir: Temporary directory used for sort.

    Returns:
        out_file: Output file path.
    """
    logger.info(f'Shuffle file {in_file}. The output file is saved in {out_file}.')
    tmp_dir = os.path.join(temp_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    cmd_template = "cat {} | awk 'BEGIN{{srand();}} {{printf \"%0.15f\\t%s\\n\", rand(), $0;}}'" \
                   " | sort -T {} -n | cut  -f 2- > {}"
    run_command(cmd_template.format(in_file, tmp_dir, out_file))
    return out_file


def shuffle_big_data(in_file, out_file, buffer_size=100):
    """Shuffle large data with buffer.

    Args:
        in_file: Input file path.
        out_file: Output file path.
        buffer_size: Size of buffered block. The default is 100.

    Returns:
        out_file: Output file path.
    """
    def open_buf(index):
        return open('.buffer/' + index, 'w', encoding='utf-8')

    if os.path.exists('.buffer'):
        shutil.rmtree('.buffer')
    if os.path.exists(out_file):
        os.remove(out_file)
    os.mkdir('./.buffer')
    random.seed(2)
    arr = [open_buf(str(x)) for x in range(buffer_size)]
    with open(in_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            arr[random.randint(0, buffer_size - 1)].write(line)
    for f in arr:
        f.close()
    for i in range(buffer_size):
        if os.path.getsize('.buffer/' + str(i)) < 1024 ** 3:
            os.system(f'shuf {"./.buffer/" + str(i)} >> {out_file}')
        else:
            os.system(f'cat {"./.buffer/" + str(i)} >> {out_file}')
    shutil.rmtree('.buffer')
    # os.remove(in_file)
    return out_file


def shuffle_small_data(in_f, out_f):
    """Shuffle small data by calling the Unix function `shuf`.

    Args:
        in_f: Input file path.
        out_f: Output file path.

    Returns:
        out_f: Output file path.
    """
    run_command(f'shuf {in_f} > {out_f}')
    return out_f


def split_data_percent(in_file, out_file1, out_file2, out_file1_percent, token_len=None):
    """Split in_file into out_file1 and out_file2 according to provided out_file1 percentage.

    Args:
        in_file: Input file.
        out_file1: Output file 1.
        out_file2: Output file 2.
        out_file1_percent: Percentage of out_file1.
        token_len: filter lines by specified length of tokens (None to disable).

    Returns:
        Count of lines in in_file, out_file1 and out_file2.
    """
    logger.info(f'Splitting file {in_file}.')
    with open(out_file1, 'w', encoding='utf-8') as fout1, \
            open(out_file2, 'w', encoding='utf-8') as fout2, \
            open(in_file, 'r', encoding='utf-8') as fin:
        total_cnt = 0
        out_file1_cnt = 0
        out_file2_cnt = 0
        for line in fin:
            if token_len is None or len(line.rstrip('\n').split('\t')) == token_len:
                total_cnt += 1
                if random.random() < out_file1_percent:
                    fout1.write(line)
                    out_file1_cnt += 1
                else:
                    fout2.write(line)
                    out_file2_cnt += 1
    logger.info(f'Splitting finish. total:{total_cnt}, {out_file1}: {out_file1_cnt}, {out_file2}: {out_file2_cnt}.')
    return total_cnt, out_file1_cnt, out_file2_cnt


def tar_files(tar_name, file_list):
    """Packaging files to tar_name.

    Args:
        tar_name: output file path,
        file_list: input file list.

    Returns:
        None.
    """
    logger.info(f'Tar file: {file_list} to {tar_name}.')
    tar = tarfile.open(tar_name, 'w:gz')
    for file in file_list:
        tar.add(file)
    tar.close()
