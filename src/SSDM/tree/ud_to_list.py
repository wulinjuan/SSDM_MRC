# -*- coding: utf-8 -*
def generate_lines_for_sent(lines):
    '''Yields batches of lines describing a sentence in conllx.

    Args:
      lines: Each line of a conllx file.
    Yields:
      a list of lines describing a single sentence in conllx.
    '''
    buf = []
    for line in lines:
        if line.startswith('#'):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield buf


def load_conll_dataset(filepath):
    '''Reads in a conllx file; generates Observation objects

    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
      filepath: the filesystem path to the conll dataset

    Returns:
      A list of Observations
    '''
    sentence_trees = []

    lines = (x for x in open(filepath, 'r', encoding='utf-8'))
    for buf in generate_lines_for_sent(lines):
        conllx_lines = []
        for line in buf:
            if "-" in line.strip().split("\t")[0] and "_	_	_	_" in line or "." in line.strip().split("\t")[0]:
                continue
            conllx_lines.append(line.strip().split('\t'))
        sentence_trees.append(conllx_lines)
    return sentence_trees
