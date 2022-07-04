import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--name', type=str, required=True)
# Parse the argument
args = parser.parse_args()
# Print "Hello" + the user input argument
# print('Hello,', args.name)

abs_path = "../"
# Read token list file
token_list = []
with open(abs_path+"models/tokens_list.txt", "r") as outfile:
    for line in outfile:
        token_list.append(line.strip('\n'))
        
# Read svd matrix
svd_mat_loaded = np.load(abs_path+'models/svd_matrix.npz')['a']

# Function to calculate the top10 words and plot using t-SNE
def plot_top10_words(word, token_list, svd_mat, file_name=args.name):

    def tsne_plot(results):
        words = []
        embeds = []

        for res in results:
            embeds.append(res[1])
            words.append(res[0])
        
        tsne_model = TSNE(init='pca')
        res_embeds = tsne_model.fit_transform(embeds)

        x_axis_val = []
        y_axis_val = []
        for val in res_embeds:
            x_axis_val.append(val[0])
            y_axis_val.append(val[1])
            
        plt.figure(figsize=(10, 10)) 
        for i in range(len(x_axis_val)):
            plt.scatter(x_axis_val[i],y_axis_val[i])
            plt.annotate(words[i],
                        xy=(x_axis_val[i],y_axis_val[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        plt.savefig(abs_path+"models/"+file_name+".png")
        #plt.show()
    
    word_index = token_list.index(word)
    word_embed = svd_mat[word_index]
    res = {}
    for i, embed in tqdm(enumerate(svd_mat)):
        if i!=word_index:
            res[i] = [1 - spatial.distance.cosine(svd_mat[i], word_embed), svd_mat[i]]


    results = []
    for t in sorted(res.items(), key=lambda item: item[1][0], reverse=True)[0:10]:
        results.append([token_list[t[0]], t[1][1]])

    print('Word:-', word)
    print('Words:-', end='\t')
    for res in results:
        print(res[0], end=', ')
    print()

    tsne_plot(results)
