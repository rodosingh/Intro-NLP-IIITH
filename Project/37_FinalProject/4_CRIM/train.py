import sys, os, random, time, argparse
import joblib
from itertools import cycle
from math import sqrt
from copy import deepcopy
from pyhocon import ConfigFactory
import numpy as np
import torch
from Projector import Projector, Classifier
from Evaluator import Evaluator
from utils import print_params, generate_embedder

doc = """Train model given a pickle file containing training
and dev data, as well as hyperparameter settings. Write model and log file."""

def generate_samples(things):
    """ Make generator that samples randomly from a list of things. """

    nb_things = len(things)
    shuffled_things = deepcopy(things)
    for i in cycle(range(nb_things)):
        if i == 0:
            random.shuffle(shuffled_things)
        yield shuffled_things[i]

def train_model(model, optim, train_q_embed, dev_q_embed, dev_q_cand_ids, 
                train_pairs, dev_pairs, hparams, log_path, seed):
    """Train model using negative sampling.

    Args:

    - model
    - optim: optimizer
    - train_q_embed: Embedding object for training queries, shape (nb
      train queries, dim)
    - dev_q_embed: Embedding object for dev queries, shape (nb dev
      queries, dim)
    - dev_q_cand_ids: list containing candidate ID of each dev query
      (None if it is not a candidate), used to compute MAP on dev set.
    - train_pairs: array of (query ID, hypernym ID) pairs
      for training
    - dev_pairs: array of (query ID, hypernym ID) pairs for
      validation
    - hparams: dict containing settings of hyperparameters
    - log_path: path of log file
    - seed: seed for RNG

    """

    # Extract hyperparameter settings
    nb_neg_samples = hparams["nb_neg_samples"]
    subsample = hparams["subsample"]
    max_epochs = hparams["max_epochs"]
    patience = hparams["patience"]
    batch_size = hparams["batch_size"]
    clip = hparams["clip"]

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Prepare sampling of negative examples
    candidate_ids = list(range(model.get_nb_candidates()))
    cand_sampler = generate_samples(candidate_ids)

    # Prepare subsampling of positive examples
    pos_sample_prob = {}
    if subsample:
        hyp_fd = {}
        for h_id in train_pairs[:,1]:
            if h_id not in hyp_fd:
                hyp_fd[h_id] = 0
            hyp_fd[h_id] += 1
        min_freq = min(hyp_fd.values())
        for (h_id, freq) in hyp_fd.items():
            pos_sample_prob[h_id] = sqrt(min_freq / freq)

    # Check if we're using CUDA
    if model.use_cuda:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    # Initialize training batch for query IDs, positive hypernym IDs,
    # negative hypernym IDs, positive targets, and negative targets.
    # targets. We separate positive and negative examples to compute
    # the losses separately. Note that this is a bit inefficient, as
    # we compute the query projections twice.
    batch_q = np.zeros(batch_size, 'int64')
    batch_h_pos = np.zeros((batch_size,1), 'int64')
    batch_h_neg = np.zeros((batch_size,nb_neg_samples), 'int64')
    t_pos_var = torch.ones((batch_size,1), requires_grad=False, device=device)
    t_neg_var = torch.zeros((batch_size,nb_neg_samples), requires_grad=False, device=device)

    # Prepare list of sets of gold hypernym IDs for queries in
    # training set. This is used for negative sampling.
    nb_train_queries = train_q_embed.weight.shape[0]
    train_gold_ids = [set() for _ in range(nb_train_queries)]
    nb_train_pairs = train_pairs.shape[0]
    for i in range(nb_train_pairs):
        q_id = int(train_pairs[i,0])
        h_id = int(train_pairs[i,1])
        train_gold_ids[q_id].add(h_id)

    # Prepare list of sets of gold hypernym IDs for queries in dev set
    # to compute score (MAP)
    nb_dev_queries = dev_q_embed.weight.shape[0]
    dev_gold_ids = [set() for _ in range(nb_dev_queries)]
    nb_dev_pairs = dev_pairs.shape[0]
    for i in range(nb_dev_pairs):
        q_id = int(dev_pairs[i,0])
        h_id = int(dev_pairs[i,1])
        dev_gold_ids[q_id].add(h_id)


    # Prepare input variables to compute loss on dev set
    dev_q_ids = torch.tensor(dev_pairs[:,0], dtype=torch.int64, device=device)
    dev_q_var = dev_q_embed(dev_q_ids)
    dev_h_var = torch.tensor(dev_pairs[:,1], dtype=torch.int64, requires_grad=False, device=device).unsqueeze(1)
    dev_t_var = torch.ones((nb_dev_pairs,1), dtype=torch.float32, requires_grad=False, device=device)

    # Make Evaluator to compute MAP on dev set
    dev_eval = Evaluator(model, dev_q_embed, dev_q_cand_ids)

    print("\nEvaluating untrained model on dev set...")
    MAP = dev_eval.get_MAP(dev_gold_ids)
    print("MAP: {:.4f}".format(MAP))

    checkpoint_header = ["Epoch", "Updates", "PosLoss", "NegLoss", 
                         "DevLoss", "DevMAP", "TimeElapsed"]
    with open(log_path, "w") as f:
        f.write("\t".join(checkpoint_header) + "\n")

    # Train model and save it if showing better performance...
    best_model = deepcopy(model)
    best_score = float("-inf")
    nb_no_gain = 0
    batch_row_id = 0
    done = False
    start_time = time.time()
    print("\nStarting training...\n")
    print("\t".join(checkpoint_header))
    for epoch in range(1,max_epochs+1):
        model.train()
        np.random.shuffle(train_pairs)
        total_pos_loss = 0.0
        total_neg_loss = 0.0

        # Loop through training pairs
        nb_updates = 0
        for pair_ix in range(train_pairs.shape[0]):
            q_id = train_pairs[pair_ix,0]
            h_id = train_pairs[pair_ix,1]
            if subsample and random.random() >= pos_sample_prob[h_id]:
                continue
            batch_q[batch_row_id] = q_id
            batch_h_pos[batch_row_id] = h_id

            # Get negative examples
            neg_samples = []
            while len(neg_samples) < nb_neg_samples:
                cand_id = next(cand_sampler)
                if cand_id not in train_gold_ids[q_id]:
                    neg_samples.append(cand_id)
            batch_h_neg[batch_row_id] = neg_samples

            # Update on batch
            batch_row_id = (batch_row_id + 1) % batch_size
            if batch_row_id + 1 == batch_size:
                q_ids = torch.tensor(batch_q, dtype=torch.int64, requires_grad=False, device=device) 
                q_var = train_q_embed(q_ids)
                h_pos_var = torch.tensor(batch_h_pos, dtype=torch.int64, requires_grad=False, device=device)
                h_neg_var = torch.tensor(batch_h_neg, dtype=torch.int64, requires_grad=False, device=device)
                optim.zero_grad()
                pos_loss = model.get_loss(q_var, h_pos_var, t_pos_var)
                neg_loss = model.get_loss(q_var, h_neg_var, t_neg_var)
                loss = pos_loss + neg_loss
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(train_q_embed.parameters(), clip)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optim.step()
                total_pos_loss += pos_loss.item()
                total_neg_loss += neg_loss.item()
                nb_updates += 1

        # Check progress
        avg_pos_loss = total_pos_loss / (nb_updates * batch_size)
        avg_neg_loss = total_neg_loss / (nb_updates * batch_size)

        # Compute loss and MAP on dev set
        model.eval()
        dev_loss = model.get_loss(dev_q_var, dev_h_var, dev_t_var)
        avg_dev_loss = dev_loss.item() / nb_dev_pairs
        MAP = dev_eval.get_MAP(dev_gold_ids)
        checkpoint_data = []
        checkpoint_data.append(str(epoch))
        checkpoint_data.append(str(nb_updates))
        checkpoint_data.append("{:.4f}".format(avg_pos_loss))
        checkpoint_data.append("{:.4f}".format(avg_neg_loss))
        checkpoint_data.append("{:.4f}".format(avg_dev_loss))
        checkpoint_data.append("{:.4f}".format(MAP))
        checkpoint_data.append("{:.1f}s".format(time.time()-start_time))
        print("\t".join(checkpoint_data))
        with open(log_path, "a") as f:
            f.write("\t".join(checkpoint_data)+"\n")

        # Early stopping
        if MAP > best_score:
            best_score = MAP
            best_model = deepcopy(model)
            nb_no_gain = 0
        else:
            nb_no_gain += 1
        if nb_no_gain >= patience:
            print("EARLY STOP!")
            done = True            
            print("\nEvaluating best model on dev set...")
            dev_eval.set_model(best_model)
            MAP = dev_eval.get_MAP(dev_gold_ids)
            print("MAP of best model: {:.3f}".format(MAP))
        if done:
            break
    print("\nTraining finished after {} epochs".format(epoch))
    return best_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("path_data", help="path of pickle file containing train and dev data")
    parser.add_argument("path_hparams", help="path of config file containing hparam settings")
    parser.add_argument("dir_model", help="path of directory in which we write the model")
    parser.add_argument("model_number", help="number of model = checkpoint number")
    parser.add_argument("-n", "--no_gpu", action="store_true")
    parser.add_argument("-s", "--seed", type=int, required=False, help="Seed for RNG")
    args = parser.parse_args()    

    # python train.py dir-data/preprocessed_data/music.pkl hparams.conf ckpts/ckpt_music 1
    
    # Parse hyperparameter settings
    hparams = ConfigFactory.parse_file(args.path_hparams)

    # Make directory where we will save model
    if os.path.exists(args.dir_model):
        msg = "There is already something at {}".format(args.dir_model)
        print(msg)
        print("Ignoring folder construction\n")
        # raise ValueError(msg)
    else:
        os.makedirs(args.dir_model)

    # Check if we can use CUDA
    use_gpu = not args.no_gpu
    if use_gpu and not torch.cuda.is_available():
        print("WARNING: CUDA is not available.")
        use_gpu = False

    # Make sure real hyperparameters have real (float) values
    for k in ["learning_rate", "beta1", "beta2", "weight_decay", "dropout", "clip"]:
        hparams[k] = float(hparams[k])

    # Print hyperparameter settings
    print("------\nHyperparameter settings:\n------")
    for k,v in hparams.items():
        print("{}: {}".format(k,v))
    print("-------\n")

    # Load data
    print("Loading data <-- {}".format(args.path_data))
    data = joblib.load(args.path_data)
    print("Data:")
    for k,v in data.items():
        print("- {} ({}.{})".format(k, type(v).__module__, type(v).__name__))
    train_pairs = data["train_pairs"]
    dev_pairs = data["dev_pairs"]
    candidates = data["candidates"]
    train_q_cand_ids = data["train_query_cand_ids"]
    dev_q_cand_ids = data["dev_query_cand_ids"]

    # Make embedders for candidates, train queries, and dev queries. The
    # query embeddings are not tuned.
    cand_embed = generate_embedder(data["candidate_embeds"], grad=True, 
                               cuda=use_gpu, sparse=False)
    train_q_embed = generate_embedder(data["train_query_embeds"], grad=True, 
                                  cuda=use_gpu, sparse=False)
    dev_q_embed = generate_embedder(data["dev_query_embeds"], grad=False, 
                                cuda=use_gpu, sparse=False)

    # Initialize model
    print("\nInitializing model...")
    projector = Projector(cand_embed, hparams["nb_maps"], hparams["dropout"], 
                          hparams["normalize_e"], hparams["normalize_p"], 
                          cuda=use_gpu, seed=args.seed)
    classifier = Classifier(projector, cuda=use_gpu, seed=args.seed)

    # Print parameter info
    print("Model parameters:")
    print_params(classifier)

    # Initialize optimizer
    lr = hparams["learning_rate"]
    betas = (hparams["beta1"], hparams["beta2"])
    wd = hparams["weight_decay"]
    trainables = list(filter(lambda x:x.requires_grad, classifier.parameters()))
    trainables.append(train_q_embed.weight)
    optim = torch.optim.Adam(trainables, lr=lr, betas=betas, eps=1e-8, weight_decay=wd)

    # Train model
    log_path = "{}/log_{}.txt".format(args.dir_model, args.model_number)
    
    model = train_model(classifier, optim, train_q_embed, dev_q_embed, dev_q_cand_ids, 
                        train_pairs, dev_pairs, hparams, log_path, args.seed)
    print("\nLog saved ---> {}".format(log_path))

    # Save model
    path = "{}/model_{}.pt".format(args.dir_model, args.model_number)
    print("Saving model ---> {}".format(path))
    torch.save(model, path)
