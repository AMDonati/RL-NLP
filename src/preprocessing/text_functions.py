#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<PAD>: Extra parts of the sequence that we should ignore
<SOS>: Goes at the start of a sequence
<EOS>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
}


def tokenize(s, add_start_token, add_end_token, punct_to_keep, punct_to_remove, delim=' '):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    start_token_upper = tokens[0]
    tokens = [w.lower() for w in tokens]  # lower all letters.
    if add_start_token:
        tokens.insert(0, '<SOS>')
    if add_end_token:
        tokens.append('<EOS>')
    return tokens, start_token_upper


def build_vocab(sequences, min_token_count, punct_to_keep, punct_to_remove, delim=' '):
    token_to_count = {}
    start_tokens = []
    for seq in sequences:
        seq_tokens, start_token_upper = tokenize(s=seq,
                                                 delim=delim,
                                                 punct_to_keep=punct_to_keep,
                                                 punct_to_remove=punct_to_remove,
                                                 add_start_token=False,
                                                 add_end_token=False)

        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1
        start_tokens.append(start_token_upper)

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    # getting the unique starting words.
    start_tokens = list(set(start_tokens))

    return token_to_idx, start_tokens


def encode(seq_tokens, token_to_idx, allow_unk):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, stop_at_end, delim=' ', clean=False, ignored=["<SOS>", "PAD"]):
    tokens = []
    for idx in seq_idx:
        token = idx_to_token[idx]
        if not (clean and token in ignored ):
            if stop_at_end and token == '<EOS>':
                break
            tokens.append(token)
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)
