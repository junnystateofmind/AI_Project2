import os
import urllib.request
import ssl
from pathlib import Path
import zipfile
import rarfile


def download_file(url, download_path):
    if not os.path.exists(download_path):
        print(f"Downloading {url} to {download_path}...")
        # SSL 인증서 검증 무시 설정
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, download_path)
        print("Download complete.")


def extract_rar(rar_path, extract_path):
    if not os.path.exists(extract_path):
        print(f"Extracting {rar_path} to {extract_path}...")
        with rarfile.RarFile(rar_path) as rar:
            rar.extractall(path=extract_path)
        print("Extraction complete.")


def extract_zip(zip_path, extract_path):
    if not os.path.exists(extract_path):
        print(f"Extracting {zip_path} to {extract_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")


def prepare_ucf101_dataset(data_dir):
    ucf101_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    annotations_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    data_path = Path(data_dir)
    raw_data_path = data_path / 'raw'
    raw_data_path.mkdir(parents=True, exist_ok=True)

    ucf101_download_path = raw_data_path / 'UCF101.rar'
    annotations_download_path = raw_data_path / 'UCF101TrainTestSplits-RecognitionTask.zip'

    ucf101_extract_path = raw_data_path / 'UCF101'
    annotations_extract_path = raw_data_path / 'annotations'

    download_file(ucf101_url, ucf101_download_path)
    extract_rar(ucf101_download_path, ucf101_extract_path)

    download_file(annotations_url, annotations_download_path)
    extract_zip(annotations_download_path, annotations_extract_path)


if __name__ == "__main__":
    data_dir = './data'
    prepare_ucf101_dataset(data_dir)
