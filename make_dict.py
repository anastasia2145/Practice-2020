import os


path = "/home/amr/fastq_files/trimmed_paired/jellyfish_result_2/"
filenames = open("data/train_files.txt", encoding="utf8").read().split("\n")[:-1]
# print(filenames)
all_kmers = set()
for i, file in enumerate(filenames):
    data = open(path + file, encoding="utf8").read().split("\n")
    values = data[::2]
    values = values[:-1]
    kmers = data[1::2]
    assert(len(values) == len(kmers))
    values = list(map(lambda x: int(x.replace(">","")), values))
    for kmer in kmers:
        all_kmers.add(kmer)
    print(i, file)
    print(len(all_kmers))
    print("---------------")
    

all_kmers = sorted(list(all_kmers))
kmer_dictionary = {kmer: i for i, kmer in enumerate(all_kmers)}       
        
with open("data/rows.txt", "w") as f:
    for filename in filenames:
        f.write(filename)
        f.write("\n")
        
with open("data/columns.txt", "w") as f:
    for kmer in kmer_dictionary.keys():
        f.write(kmer)
        f.write("\n")

