import re
import time
import numpy as np
import pandas as pd

path_ = "/home/amr/fastq_files/trimmed_paired/jellyfish_result_2/"
path = "/home/amr/fastq_files/trimmed_paired/jellyfish_result/cut/"

kmers = open("data/columns.txt").read().split()
files = open("/home/amr/rows.txt").read().split()
     
kmers_dictionary = {kmer: i for i, kmer in enumerate(kmers)}
matrix = np.zeros((len(files), len(kmers)))

for i, file in enumerate(files):
    start = time.time()
    data = open(path + file, encoding="utf8").read().split("\n")
    values = data[::2]
    current_values = values[:-1]
    current_kmers = data[1::2]
    assert(len(current_values) == len(current_kmers))
    current_values = list(map(lambda x: int(x.replace(">","")), current_values))
    for k, kmer in enumerate(current_kmers):
        j = kmers_dictionary[kmer]
        matrix[i][j] = current_values[k]
    print(i, file)
    print(time.time() - start)
    
    
klebseilla_gentamicin = "data/Klebseilla_gentamicin.csv"
data = pd.read_csv(klebseilla_gentamicin)
# short_data = data[["Antibiotic", "Measurement.Sign", "Measurement.Value", "SRA.Accession"]]

srr_names = []
for file in files:
    srr_names += re.findall("SRR\d+", file)
srr_names = list(set(srr_names))

data = data.loc[data['SRA.Accession'].isin(srr_names)]
srr2mic = {srr : mic for srr, mic in zip(list(data['SRA.Accession']), list(data['Measurement.Value']))}

identificators = [file[:10] for i, file in enumerate(files)]
mic = np.array([srr2mic[ind] for ind in identificators])
mic = mic.reshape((-1, 1))


# temp = np.copy(matrix[:, :-1])
# temp[temp > 0] = 1
# binary_matrix = np.hstack((temp, mic))

count_matrix = np.hstack((matrix, mic))

np.savetxt("data/matrix_test.csv", count_matrix, delimiter=",", fmt='%d')
np.savetxt("data/matrix_test.tsv", count_matrix, delimiter="\t", fmt='%d')
# np.savetxt("binary_matrix_.csv", binary_matrix, delimiter=",", fmt='%d')

np.save("data/matrix_test.npy", count_matrix)
