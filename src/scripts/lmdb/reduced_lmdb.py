import pickle

import lmdb

def save(reduced_db_txn, path='data/datasets/flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb'):
    db = lmdb.open(path)
    db_txn = db.begin(write=False)
    cursor = db_txn.cursor()

    i = 0
    for key, value in cursor:
        i += 1
        reduced_db_txn.put(key, value)
        reduced_db_txn.commit()
        if i > 10:
           break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lmdb_path", type=str, default='data/datasets/flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb',
                        help="data folder for the full lmdb")
    parser.add_argument("-reduced_path", type=str, default="../vilbert-multi-task/data/datasets/flickr30k/reduced.lmdb")
    args = parser.parse_args()
    reduced_db = lmdb.open(args.reduced_path)

    reduced_db_txn = reduced_db.begin(write=True)

    save(reduced_db_txn, args.lmdb_path)

    reduced_db_txn.put("keys".encode(), pickle.dumps([key for key, _ in reduced_db_txn.cursor()]))
    reduced_db_txn.commit()
    reduced_db.close()
