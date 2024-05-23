import os
import requests
import zipfile
from tqdm import tqdm
import rarfile


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)


def extract_rar(rar_path, extract_path):
    with rarfile.RarFile(rar_path) as rar:
        rar.extractall(path=extract_path)
    print(f"Extracted {rar_path} to {extract_path}")


def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted {zip_path} to {extract_path}")


def prepare_ucf101_dataset(data_dir):
    ucf101_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    annotations_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    raw_data_path = os.path.join(data_dir, 'raw')
    os.makedirs(raw_data_path, exist_ok=True)

    ucf101_rar_path = os.path.join(raw_data_path, 'UCF101.rar')
    annotations_zip_path = os.path.join(raw_data_path, 'UCF101TrainTestSplits-RecognitionTask.zip')

    ucf101_extract_path = os.path.join(raw_data_path, 'UCF101')
    annotations_extract_path = os.path.join(raw_data_path, 'annotations')

    if not os.path.exists(ucf101_rar_path):
        download_file(ucf101_url, ucf101_rar_path)
    if not os.path.exists(annotations_zip_path):
        download_file(annotations_url, annotations_zip_path)

    if not os.path.exists(ucf101_extract_path):
        extract_rar(ucf101_rar_path, ucf101_extract_path)
    if not os.path.exists(annotations_extract_path):
        extract_zip(annotations_zip_path, annotations_extract_path)


if __name__ == "__main__":
    data_dir = '.'
    prepare_ucf101_dataset(data_dir)
