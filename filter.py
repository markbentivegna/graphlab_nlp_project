import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser(description="Filter HDFS logs")
parser.add_argument("hdfs_file_location", metavar="h", type=str, help="File location for HDFS log file")
parser.add_argument("anamoly_file_location", metavar="a", type=str, help="File location for anomaly CSV file")
parser.add_argument("output_file_location", metavar="o", type=str, help="Output file location for truncated HDFS logs")
parser.add_argument("blocks_count", metavar="n", type=int, help="Number of HDFS blocks to select for training")
parser.add_argument("shuffle", metavar="s", type=bool, help="Shuffle dataset before selecting HDFS blocks")
args = parser.parse_args()

LOG_FILE = args.hdfs_file_location
ANOMALY_FILE = args.anamoly_file_location
OUTPUT_FILE = args.output_file_location
blocks_count = args.blocks_count
shuffle = args.shuffle

block_ids = pd.read_csv(ANOMALY_FILE)["BlockId"].to_list()
if shuffle:
    random.shuffle(block_ids)

block_ids = block_ids[:blocks_count]

def validate_block_id(line):
    for block_id in block_ids:
        if block_id in line:
            return True
    return False

log_lines = []
with open(LOG_FILE) as f:
    for line in f:
        if validate_block_id(line):
            log_lines.append(line)

with open(OUTPUT_FILE, "w") as f:
    for line in log_lines:
        f.write(line)