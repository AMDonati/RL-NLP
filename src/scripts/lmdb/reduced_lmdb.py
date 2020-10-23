import pickle

import lmdb

def save(path='data/datasets/flickr30k/flickr30k_resnext152_faster_rcnn_genome.lmdb'):
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

    reduced_db = lmdb.open("../vilbert-multi-task/data/datasets/flickr30k/reduced.lmdb")


    reduced_db_txn = reduced_db.begin(write=True)

    #save(path, reduced_db_txn)

    reduced_db_txn.put("keys".encode(), pickle.dumps([key for key, _ in reduced_db_txn.cursor()]))
    reduced_db_txn.commit()
    reduced_db.close()

    #cursor = reduced_db_txn.cursor()

    #for key, value in cursor:
        #print(key)
