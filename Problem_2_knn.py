from functions import io
from Problem_1_CosineSimilarity import calculate_cosine
from heapq import heappush, heappop
import argparse

def get_k_nearest_words(model, word, k):
    if word not in model:
        return ['Word does not exist']
    else:
        h = []
        vec1 = model[word]
        for w in model:
            if w != word:
                vec2 = model[w]
                dist = calculate_cosine(vec1, vec2)
                heappush(h, (dist, w))
        result = []
        for i in range(k):
            result.append(heappop(h)[1])
        return result

def main():
    model = io.load_w2v('word2vec/W2V_150.txt')

    # pre handle input
    parser = argparse.ArgumentParser(description='Problem 2 K-nearest words')
    parser.add_argument('-k', help='input k', type=int, default=10, required=False)
    parser.add_argument('-w', help='input word', type=str, default='sinh_viên', required=False)
    opt = parser.parse_args()
    print('Running opt: {}'.format(opt))

    try:
        k_cosine = get_k_nearest_words(model, opt.w, opt.k)
        output = []
        output.append('word: {}'.format(opt.w))
        output.append('{} nearest words: {}'.format(opt.k, k_cosine))
        print('word:', opt.w)
        print(opt.k, 'nearest words:', k_cosine)
        io.write_output("Problem_2_result.txt", output)
    except:
        print('word is not exist')

if __name__ == '__main__':
    main()
