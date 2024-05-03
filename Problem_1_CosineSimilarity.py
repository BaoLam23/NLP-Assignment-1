from scipy import stats
import pandas as pd
from functions import io
import math

def calculate_cosine(v1, v2):
    return 1.0*(sum(i*j for i,j in zip(v1,v2)))/(math.sqrt(sum(i**2 for i in v1)) + math.sqrt(sum(j**2 for j in v2)))

def calculate_similarity(input_file, model):
    output = []
    rs = []
    ss = []
    dataset = open(input_file, 'r', encoding='utf-8')
    dataset.readline()
    for i in dataset:
        s = i.split()
        if (s[0].strip() in model and s[1].strip() in model):
            v1 = model[s[0].strip()] # word 1
            v2 = model[s[1].strip()] # word 2
            distance_cosine = calculate_cosine(v1, v2)

            pcc = stats.pearsonr(v1, v2)
            src = stats.spearmanr(v1, v2)

            sim = (2 - distance_cosine) / 2
            rs.append(sim)
            sim1 = (float(s[4]) / 10)
            ss.append(sim1)

            output.append((s[0], s[1], distance_cosine, pcc, src))
    pearsonr = stats.pearsonr(rs, ss)
    spearmanr = stats.spearmanr(rs, ss)
    return output, pearsonr, spearmanr

def main():
    model = io.load_w2v('word2vec/W2V_150.txt')
    vsim_400_file = 'datasets/ViSim-400/Visim-400.txt'
    output, pearsonr, spearmanr = calculate_similarity(vsim_400_file, model)
    df = pd.DataFrame(columns=['Word1', 'Word2', 'Cosine', 'Pearson CC', 'Spearman\'s rank CC'], data=output)
    df.to_csv('Problem_1_result.csv', encoding='utf-8', header=True)
    print(" Pearson correlation coefficient: ", pearsonr)
    print(" Spearman's rank correlation coefficient: ", spearmanr)

if __name__ == '__main__':
    main()
