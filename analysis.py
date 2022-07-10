import pandas as pd
import json
import ast
import collections
import matplotlib.pyplot as plt

normal_file = "normal.csv"
anomaly_file = "anomaly.csv"

csv_path = "output/hdfs/"

normal_df = pd.read_csv(f"{csv_path}{normal_file}")
anomaly_df = pd.read_csv(f"{csv_path}{anomaly_file}")
normal_list = normal_df.sequence.apply(ast.literal_eval).explode("sequence").tolist()
anomaly_list = anomaly_df.sequence.apply(ast.literal_eval).explode("sequence").tolist()
normal_counter=collections.Counter(normal_list)
anomaly_counter=collections.Counter(anomaly_list)

# plt.bar(normal_counter.keys(), normal_counter.values(), color='g')
# plt.show()

# plt.bar(anomaly_counter.keys(), anomaly_counter.values(), color='r')
# plt.show()

sorted_normal_list = sorted(normal_counter.items())
sorted_anomaly_list = sorted(anomaly_counter.items())

def json_dump(sorted_list, filename):
    json_obj = {}
    for item in sorted_list:
        json_obj[item[0]] = item[1]
    with open(filename, 'w') as f:
        json.dump(json_obj, f, indent=4)

json_dump(sorted_normal_list, "normal_logs_count.json")
json_dump(sorted_anomaly_list, "anomaly_logs_count.json")

anomaly_keys = list(anomaly_counter.keys() - normal_counter.keys())
normal_keys = list(normal_counter.keys())

selected_vals = [21, 20, 18, 15, 19, 11, 12, 13, 14]
for val in selected_vals:
    anomaly_keys.append(val)
    normal_keys.remove(val)

tp = 0
for row in anomaly_df["sequence"].items():
    if len(list(set(ast.literal_eval(row[1])) & set(anomaly_keys))) > 0:
        tp += 1

tn = 0
for row in normal_df["sequence"].items():
    if len(list(set(ast.literal_eval(row[1])) & set(anomaly_keys))) == 0:
        tn += 1

fn = anomaly_df.shape[0] - tp
fp = normal_df.shape[0] - tn

recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2*((precision * recall) / (precision + recall))
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"PRECISION: {precision} RECALL: {recall} F1: {f1} ACCURACY: {accuracy}")
