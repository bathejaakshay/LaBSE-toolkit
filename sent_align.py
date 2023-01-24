from sentence_transformers import SentenceTransformer
import torch
import math
from labse_score import get_score
import numpy as np
import argparse
from tqdm import tqdm
import os

"""
HELP-OPTIONS:

-src, --source: PATH to source file
-tgt, --target: PATH to target file
-th, --threshold: LaBSE threshold value for extracting high quality data
-b, --batch_size: batch_size for LaBSE scoring 
-op, --operation: Select operation between score and sent-align
-mp, --model_path: Path to the saved LaBSE model

"""

class LaBSE:
    """
    A multilingual embedding model is a powerful tool that encodes text from different languages into a shared embedding space, enabling it to be applied to a range of downstream tasks, like text classification, clustering, and others, while also leveraging semantic information for language understanding. 
    """
    model_labse = None
    device = None
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Loading Model')
        if not model_path:
            self.model_labse = SentenceTransformer('sentence-transformers/LaBSE')
        else:
            self.model_labse = SentenceTransformer(model_path)
        self.model_labse.to(self.device)
        print('Done')
    
    def sent_align(self, src, tgt, threshold, batch_size):
        """
        Align Sentence pairs in the misaligned parallel corpus.

        src: List of sentences in source language
        tgt: List of sentences in target language
        threshold: Float Threshold value for extracting high quality parallel sentences
        batch_size: Int Batch-size for running LABSE script. 

        output:
        Saves correctly aligned parallel sentences as per threshold value.
        """
        final_src=[]
        final_tgt=[]
        for i in tqdm(range(len(src)), position=0, leave=True) as pbar:
            temp_src = [src[i]]*len(tgt)
            temp_tgt = tgt
            scores = get_score(temp_src, temp_tgt, self.model_labse, batch_size)
            # print(f'{max(scores["similarity"])} , {threshold}')
            if max(scores["similarity"]) >= threshold:
                ind = np.where(scores["similarity"] == max(scores["similarity"]))[0][0]
                final_src.append(temp_src[ind])
                final_tgt.append(temp_tgt[ind])
            pbar.update()
        path = os.getcwd()
        path+="/out"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+"/output_src.txt","w") as f:
            f.writelines(final_src)
        with open(path+"/output_tgt.txt","w") as f:
            f.writelines(final_tgt)
        return

    def score(self, src, tgt, threshold, batch_size):
        """
        Generating LaBSE scores for the given parallel corpus

        src: List of sentences in source language
        tgt: List of sentences in target language
        threshold: Float Threshold value for extracting high quality parallel sentences
        batch_size: Int Batch-size for running LABSE script. 

        output:
        Saves scores in the list if threshold value is None/ Not passed as flag
        Saves extracted parallel sentences if threshold value is passed.
        """
        final_src = []
        final_tgt = []
        score = get_score(src, tgt, self.model_labse, batch_size)
        score_n = score["similarity"].to_list()
        scores = [str(x) + '\n' for x in score_n]
        if not threshold:
            path = os.getcwd()
            path+="/out"
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path+"/output_scores.txt","w") as f:
                f.writelines(scores)

        else:
            for i in range(len(scores)):
                if score_n[i] > threshold:
                    final_src.append(src[i])
                    final_tgt.append(tgt[i])

            path = os.getcwd()
            path+="/out"
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path+"/output_src.txt","w") as f:
                f.writelines(final_src)
            with open(path+"/output_tgt.txt","w") as f:
                f.writelines(final_tgt)
              
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-src","--source", help = "path to source file")
    parser.add_argument("-tgt","--target", help = "path to target file")
    parser.add_argument("-lp","--model_path",nargs = "?", const = None,  help = "Path to LaBSE model")
    parser.add_argument("-th", "--threshold", nargs = '?', const = None , help = "LaBSE Threshold", type=float)
    parser.add_argument("-b", "--batch_size", nargs = '?', const = 1000, help = "batch size", type=int)
    parser.add_argument("-op", "--operation", nargs = '?', const = "score", help = "Select operation scoring / sentence alignment")
    args = parser.parse_args()
    src_path = args.source
    tgt_path = args.target
    model_path = args.model_path
    threshold = args.threshold
    batch_size = args.batch_size
    operation = args.operation
    print("args: threshold = " +  str(threshold) + " batch_size = " + str(batch_size) + " operation = " + operation) 
    with open(src_path, "r") as f, open(tgt_path, "r") as f1:
        src = f.readlines()
        tgt = f1.readlines()
    lab_ob = LaBSE(model_path)
    print("LaBSE model loaded")
    try:
        if operation == "score":
            lab_ob.score(src, tgt, threshold, batch_size)
        elif operation == "sent-align":
            lab_ob.sent_align(src, tgt, threshold, batch_size) 
        else:
            raise ValueError("Not a valid operation, select between score and sent-align")
    except Exception as err:
        print("ERROR: " + repr(err))

    print("completed")