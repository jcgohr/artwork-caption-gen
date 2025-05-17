import csv
import argparse

"""
A script used to find the largest MRR (RR in this case) drop between retrieval models.
"""

def read_csv(file_path):
    data = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)[1:]  # extract query IDs
        for row in reader:
            metric = row[0]
            values = list(map(float, row[1:]))
            data[metric] = dict(zip(headers, values))
    return data

def get_top_n_mrr_drops(file1, file2, top_n):
    data1 = read_csv(file1)
    data2 = read_csv(file2)
    
    if 'mrr' not in data1 or 'mrr' not in data2:
        raise ValueError("Both files must contain an 'mrr' row.")
    
    mrr_diff = {qid: data1['mrr'][qid] - data2['mrr'][qid] for qid in data1['mrr']}
    sorted_drops = sorted(mrr_diff.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_drops[:top_n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top N query IDs with largest MRR drops.")
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")
    parser.add_argument("top_n", type=int, help="Number of results to return")
    
    args = parser.parse_args()
    result = get_top_n_mrr_drops(args.file1, args.file2, args.top_n)
    print("Top query IDs with largest MRR drops:")
    for qid, drop in result:
        print(f"Query ID: {qid}, Drop Size: {drop:.4f}")