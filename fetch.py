import urllib.request
import tarfile
import argparse

parser = argparse.ArgumentParser(description="Fetch HDFS dataset")
parser.add_argument("hdfs_file_url", metavar="h", type=str, help="URL for HDFS tar file")
parser.add_argument("tar_file_location", metavar="t", type=str, help="File location for HDFS tar file")
parser.add_argument("output_directory", metavar="o", type=str, help="Output directory for HDFS tar file")
args = parser.parse_args()

HDFS_1_URL = args.hdfs_file_url
TAR_FILE_LOCATION = args.tar_file_location
OUTPUT_DIRECTORY = args.output_directory

urllib.request.urlretrieve(HDFS_1_URL, TAR_FILE_LOCATION)

file = tarfile.open(TAR_FILE_LOCATION)

file.extractall(OUTPUT_DIRECTORY)
file.close()
