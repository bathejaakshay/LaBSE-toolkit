from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import argparse
import math
import pandas as pd
# labse setup
def normalization(embeds):
    norms = torch.linalg.norm(embeds, 2, 1, keepdims=True)
    return embeds/norms

def get_sim_scores(sentences_src, sentences_tgt, model_labse):
  
    embeddings_labse_src = torch.from_numpy(model_labse.encode(sentences_src))
    embeddings_labse_tgt = torch.from_numpy(model_labse.encode(sentences_tgt))

    embeddings_labse_src = normalization(embeddings_labse_src)
    embeddings_labse_tgt = normalization(embeddings_labse_tgt)

    #computing cosine similarity scores
    labse_score = torch.matmul(embeddings_labse_src, embeddings_labse_tgt.T)

    scores = [float(labse_score[i][i]) for i in range(len(labse_score))]

    return scores


def get_score(src_sents, tgt_sents, model_labse, batch_size, pbar = True):
    clean_src_sents, clean_tgt_sents = [], []
    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        str_src_sent = str(src_sent).strip()
        str_tgt_sent = str(tgt_sent).strip()
        if len(str_src_sent.split(" ")) < 1 or len(str_tgt_sent.split(" ")) < 1:
            continue
        else:
            clean_src_sents.append(str_src_sent)
            clean_tgt_sents.append(str_tgt_sent)
    
    num_batches = math.ceil(len(clean_src_sents) / batch_size)
    scores = []
    if pbar:
        for i in tqdm(range(num_batches)):
            src_sents = clean_src_sents[i * batch_size : i * batch_size + batch_size]
            tgt_sents = clean_tgt_sents[i * batch_size : i * batch_size + batch_size]
            temp_scores = get_sim_scores(src_sents, tgt_sents, model_labse)
            scores.extend(temp_scores)
    else:
        for i in range(num_batches):
            src_sents = clean_src_sents[i * batch_size : i * batch_size + batch_size]
            tgt_sents = clean_tgt_sents[i * batch_size : i * batch_size + batch_size]
            temp_scores = get_sim_scores(src_sents, tgt_sents, model_labse)
            scores.extend(temp_scores)
    final_scores = pd.DataFrame({"similarity":scores})
    return final_scores
    # if '/' in src_file_path:
    #     src_op_path = src_file_path[:src_file_path.rfind('/') + 1] + 'filtered_' + src_file_path[src_file_path.rfind('/') + 1:]
    #     tgt_op_path = tgt_file_path[:tgt_file_path.rfind('/') + 1] + 'filtered_' + tgt_file_path[tgt_file_path.rfind('/') + 1:]
    #     scores_file_path = src_file_path[:src_file_path.rfind('/') + 1] + 'similarity-' + str(num) + '.sc'
    # else:
    #     src_op_path = 'filtered_' + src_file_path
    #     tgt_op_path = 'filtered_' + tgt_file_path
    #     scores_file_path = 'similarity-' + str(num) + '.txt'


    # print('Storing Scores')
    # with open(scores_file_path, 'w', encoding='utf-8') as f:
    #     for score in scores:
    #         f.write(str(score) + '\n')
    # print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help = "Source File Name")
    parser.add_argument("-t", "--target", help = "Target File Name")
    parser.add_argument("-n", "--number", help = "File Number")
    parser.add_argument("-b", "--batch_size", help="Batch Size")
    args = parser.parse_args()
    src_file_path = args.source
    tgt_file_path = args.target
    num = int(args.number)
    batch_size = int(args.batch_size)

    


#print('Loading Model')
#model_labse = SentenceTransformer('sentence-transformers/LaBSE')
#model_labse.to(device)
#print('Done')

