import sys, os, codecs, joblib, argparse
import numpy as np
import utils

# Do we remove training and dev queries that don't have a pre-trained
# embedding? If not, we assign a random embedding.
REMOVE_OOV_QUERIES = False
# REMOVE_OOV_DEV_QUERIES = True

doc = """ Prepare data to train and test a model on a given dataset,
and write in a pickle file. """
    
def check_queries_embedding(q_set, h_set, embed_vocab_set, mode):
    """Check for queries that don't have a pre-trained embedding"""

    print(f"Checking for {mode} queries that don't have a pre-trained embedding...")
    oov_query_ix_train = [i for i,q in enumerate(q_set) if q not in embed_vocab_set]
    print(f"Nb {mode} queries without a pre-trained embedding: {len(oov_query_ix_train)}")
    if len(oov_query_ix_train):
        m = ", ".join(q_set[i] for i in oov_query_ix_train)
        if REMOVE_OOV_QUERIES:
            print("WARNING: these training queries will be removed: {}".format(m))
            keep = sorted(set(range(len(q_set))).difference(oov_query_ix_train))
            q_set = [q_set[i] for i in keep]
            h_set = [h_set[i] for i in keep]
        else:
            print("WARNING: these training queries will be assigned a random embedding: {}".format(m))
    return q_set, h_set

def make_pairs(queries, hyps, query2id, hyp2id):
    """ Given a list of queries, a list of lists of gold hypernyms, a
    dict that maps from queries to IDs and a dict that maps from
    hypernyms to IDs, make a list of (query ID, hypernym ID)
    pairs."""
    pairs = []
    for i in range(len(queries)):
        q = queries[i]
        q_id = query2id[q]
        for h in hyps[i]:
            h_id = hyp2id[h]
            pairs.append([q_id, h_id])
    pairs = np.array(pairs, dtype=np.int64)
    return pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("subtask", choices=["1A", "1B", "1C", "2A", "2B"], help="subtask")
    msg = ("path of directory containing datasets for SemEval-2018 Task 9")
    parser.add_argument("dir_datasets", help=msg)
    msg = ("path of text file containing embeddings of the queries and candidates")
    parser.add_argument("path_embeddings", help=msg)
    parser.add_argument("path_output", help="path of output (pickle file)")
    parser.add_argument("-s", "--seed", type=int, default=91500)
    args = parser.parse_args()

    # Check dataset name
    dataset_name_exp = utils.expand_subtask_name(args.subtask)

    # Load candidates
    print("Loading candidates...")
    path_candidates = "{}/vocabulary/{}.vocabulary.txt".format(args.dir_datasets, dataset_name_exp)
    candidates = utils.load_candidates(path_candidates, normalize=True)
    print("Nb candidates: {}".format(len(candidates)))
    
    # Load queries 
    print("Loading queries...")
    path_q_train = "{}/training/data/{}.training.data.txt".format(args.dir_datasets, dataset_name_exp)
    path_q_dev = "{}/trial/data/{}.trial.data.txt".format(args.dir_datasets, dataset_name_exp)
    path_q_test = "{}/test/data/{}.test.data.txt".format(args.dir_datasets, dataset_name_exp)
    q_train, _ =  utils.load_queries(path_q_train, normalize=True)
    q_dev, _ = utils.load_queries(path_q_dev, normalize=True)
    q_test, _ =  utils.load_queries(path_q_test, normalize=True)
    print("Nb training queries: {}".format(len(q_train)))
    print("Nb dev queries: {}".format(len(q_dev)))
    print("Nb test queries: {}".format(len(q_test)))

    # Load gold hypernyms (train and dev only)
    print("Loading gold hypernyms...")
    path_h_train = "{}/training/gold/{}.training.gold.txt".format(args.dir_datasets, dataset_name_exp)
    path_h_dev = "{}/trial/gold/{}.trial.gold.txt".format(args.dir_datasets, dataset_name_exp)
    h_train = utils.load_hypernyms(path_h_train, normalize=True)
    h_dev = utils.load_hypernyms(path_h_dev, normalize=True)
    print("Nb training pairs: {}".format(sum(len(x) for x in h_train)))
    print("Nb dev pairs: {}".format(sum(len(x) for x in h_dev)))

    # Load word embeddings
    print("Loading pre-trained word embeddings...")
    embed_vocab_list, word2vec = utils.get_embeddings(args.path_embeddings, np.float32)
    embed_vocab_set = set(embed_vocab_list)
    print("Nb embeddings: {}".format(len(embed_vocab_list)))

    # Check for candidates that don't have a pre-trained emedding
    print("Checking for candidates that don't have a pre-trained embedding...")
    oov_candidates = set(c for c in candidates if c not in embed_vocab_set)
    print("Nb candidates without a pre-trained embedding: {}".format(len(oov_candidates)))
    if len(oov_candidates):
        print("WARNING: {} candidates will be assigned a random embedding.".format(len(oov_candidates)))
    
    # Check for queries that don't have a pre-trained embedding
    q_train, h_train = check_queries_embedding(q_train, h_train, embed_vocab_set, mode="training")
    q_dev, h_dev = check_queries_embedding(q_dev, h_dev, embed_vocab_set, mode="dev")
    
    print("Checking for test queries that don't have a pre-trained embedding...")
    oov_query_ix_test = [i for i,q in enumerate(q_test) if q not in embed_vocab_set]
    print("Nb test queries without a pre-trained embedding: {}".format(len(oov_query_ix_test)))
    if len(oov_query_ix_test):
        m = ", ".join(q_test[i] for i in oov_query_ix_test)
        print("WARNING: these dev queries will be assigned a random embedding: {}".format(m))

    # Make embedding arrays
    print("Making embedding array for candidates...")
    if args.subtask in ["1B", "1C"]:
        candidate_embeds = utils.make_embedding_matrix_from_pretrained(word2vec, candidates, seed=args.seed)
    else:
        candidate_embeds = utils.make_embedding_matrix(word2vec, candidates, seed=args.seed)
    candidate_embeds = utils.normalize_numpy_matrix(candidate_embeds)
    print("Nb embeddings: {}".format(candidate_embeds.shape[0]))

    print("Making embedding array for training queries...")
    if args.subtask in ["1B", "1C"]:
        train_query_embeds = utils.make_embedding_matrix_from_pretrained(word2vec, q_train, seed=args.seed)
    else:
        train_query_embeds = utils.make_embedding_matrix(word2vec, q_train, seed=args.seed)
    train_query_embeds = utils.normalize_numpy_matrix(train_query_embeds)
    print("Nb embeddings: {}".format(train_query_embeds.shape[0]))

    print("Making embedding array for dev queries...")
    if args.subtask in ["1B", "1C"]:
        dev_query_embeds = utils.make_embedding_matrix_from_pretrained(word2vec, q_dev, seed=args.seed)
    else:
        dev_query_embeds = utils.make_embedding_matrix(word2vec, q_dev, seed=args.seed)
    dev_query_embeds = utils.normalize_numpy_matrix(dev_query_embeds)
    print("Nb embeddings: {}".format(dev_query_embeds.shape[0]))

    print("Making embedding array for test queries...")
    if args.subtask in ["1B", "1C"]:
        test_query_embeds = utils.make_embedding_matrix_from_pretrained(word2vec, q_test, seed=args.seed)
    else:
        test_query_embeds = utils.make_embedding_matrix(word2vec, q_test, seed=args.seed)
    test_query_embeds = utils.normalize_numpy_matrix(test_query_embeds)
    print("Nb embeddings: {}".format(test_query_embeds.shape[0]))
    
    # Make array of (query IDs, hypernym ID) pairs
    print("Making array of (query ID, hypernym ID) pairs...")
    candidate_to_id = {w:i for i,w in enumerate(candidates)}
    train_query_to_id = {w:i for i,w in enumerate(q_train)}
    dev_query_to_id = {w:i for i,w in enumerate(q_dev)}
    train_pairs = make_pairs(q_train, h_train, train_query_to_id, candidate_to_id)
    dev_pairs = make_pairs(q_dev, h_dev, dev_query_to_id, candidate_to_id)
    print("Nb train pairs: {}".format(train_pairs.shape[0]))
    print("Nb dev pairs: {}".format(dev_pairs.shape[0]))
    
    # Check for queries that are also candidates. Make list of query
    # candidate IDs (None for queries that are not candidates)
    train_q_cand_ids = [candidate_to_id[q] if q in candidate_to_id else None for q in q_train]
    dev_q_cand_ids = [candidate_to_id[q] if q in candidate_to_id else None for q in q_dev]
    test_q_cand_ids = [candidate_to_id[q] if q in candidate_to_id else None for q in q_test]
    nb_cand_q_train = sum(1 for i in train_q_cand_ids if i is not None)
    nb_cand_q_dev = sum(1 for i in dev_q_cand_ids if i is not None)
    nb_cand_q_test = sum(1 for i in test_q_cand_ids if i is not None)
    print("Nb training queries that are also candidates: {}".format(nb_cand_q_train))
    print("Nb dev queries that are also candidates: {}".format(nb_cand_q_dev))
    print("Nb test queries that are also candidates: {}".format(nb_cand_q_test))
    
    # Pickle and dump data
    data = {}
    data["candidates"] = candidates
    data["candidate_embeds"] = candidate_embeds
    data["train_queries"] = q_train
    data["train_query_embeds"] = train_query_embeds
    data["train_query_cand_ids"] = train_q_cand_ids
    data["dev_queries"] = q_dev
    data["dev_query_embeds"] = dev_query_embeds
    data["dev_query_cand_ids"] = dev_q_cand_ids
    data["test_queries"] = q_test
    data["test_query_embeds"] = test_query_embeds
    data["test_query_cand_ids"] = test_q_cand_ids
    data["train_pairs"] = train_pairs
    data["dev_pairs"] = dev_pairs
    print("\nData:")
    for k,v in data.items():
        print("- {} ({}.{})".format(k, type(v).__module__, type(v).__name__))
    joblib.dump(data, args.path_output)
    print("\nWrote data --> {}\n".format(args.path_output))

# python prep_data.py 2A dir-data embeddings/medical_embed.txt dir-data/preprocessed_data/medical.pkl
# python prep_data.py 1B dir-data embeddings/italian_embed_300.txt dir-data/preprocessed_data/italian_300.pkl