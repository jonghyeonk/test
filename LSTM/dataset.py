import sys

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split

from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter


def add_noise(data, noise_level=0.007):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def time_warp(data, num_warps=3, warp_scale=0.2):
    signal_length = data.shape[0]
    warp_start = int(signal_length * 0.15)
    warp_end = int(signal_length * 0.85)

    warp_points = np.sort(np.random.choice(range(warp_start, warp_end), num_warps - 2, replace=False))
    warp_points = np.insert(warp_points, [0, warp_points.size], [0, signal_length - 1])
    warp_values = warp_points + np.random.uniform(-warp_scale, warp_scale, num_warps) * signal_length
    warp_values = np.clip(warp_values, 0, signal_length - 1)
    original_points = np.arange(signal_length)
    warped_signal = CubicSpline(warp_points, warp_values, bc_type='natural')(original_points)

    warped_data = data.copy()
    for i in range(data.shape[1]):
        interpolated = np.interp(original_points, warped_signal, data[:, i])
        warped_data[:, i] = interpolated if np.all(np.isfinite(interpolated)) else data[:, i]
    return warped_data


def signal_distortion(data, frequency=1.0, amplitude=0.02):
    time_steps = np.arange(data.shape[0])
    distortion = np.tile(amplitude * np.sin(2 * np.pi * frequency * time_steps / data.shape[0]), (data.shape[1], 1)).T
    return data + distortion


def filtering(data, filter_strength=2):
    filtered_data = data.copy()
    for i in range(data.shape[1]):
        filtered_data[:, i] = gaussian_filter(data[:, i], sigma=filter_strength)
    return filtered_data


def trim_eog_data(eog_data, trim_length=100):
    """
    EOG 데이터의 HEOG와 VEOG를 개별적으로 잘라낸 후 다시 결합합니다.

    Parameters:
    - eog_data: (batch_size, seq_len, 2) 형식의 데이터 (HEOG와 VEOG)
    - trim_length: 잘라낼 구간의 길이

    Returns:
    - 잘라낸 EOG 데이터
    """
    # eog_data의 첫 번째 열이 HEOG, 두 번째 열이 VEOG라고 가정
    heog_data = eog_data[:, 0]  # HEOG (750,)
    veog_data = eog_data[:, 1]  # VEOG (750,)

    # 각 신호를 개별적으로 자름
    trimmed_heog = heog_data[trim_length:-trim_length]  # (750 - 2 * trim_length,)
    trimmed_veog = veog_data[trim_length:-trim_length]  # (750 - 2 * trim_length,)

    # 자른 신호를 다시 결합 (2차원 배열로 합침)
    trimmed_eog_data = torch.stack((trimmed_heog, trimmed_veog), dim=-1)  # (seq_len - 2 * trim_length, 2)

    return trimmed_eog_data


def sliding_multi(x, size=30, wait=150, alpha=0.5):
    gap_hist = []
    save = []
    for i in range(wait, len(x) - size + 1 - 10):
        gap = 0
        window = x[i:(i + size - 1)]
        seemax = np.max(window)
        seemin = np.min(window)
        see = seemax - seemin
        save.append(see)

        if i > wait + size:
            cond = ((see > save[len(save) - 2] + alpha * np.std(save[0:(len(save) - 1)])) and (
                        np.std(save[0:(len(save) - 1)]) > 0))
            if cond:
                # amax = x[(i):(i + size - 1)].index(seemax)
                # amin = x[(i):(i + size - 1)].index(seemin)

                amax = np.argmax(window)  # 최대값 인덱스
                amin = np.argmin(window)  # 최소값 인덱스

                if amax < amin:
                    gap = -1 * see
                else:
                    gap = see

                for n in range(10):
                    window = x[(i + n + 1):(i + n + size)]
                    seemax = np.max(window)
                    seemin = np.min(window)
                    amax = np.argmax(window)
                    amin = np.argmin(window)
                    see = seemax - seemin
                    if amax < amin:
                        gap = -1 * see
                    else:
                        gap = see
        gap_hist.append(gap)

    return gap_hist


class EOGDataset(Dataset):
    def __init__(self, root='./EOG-Data', split='train', padding_length=751, augmentation=False, args=None):
        self.root = os.path.join(root, split)

        self.csv_list = os.listdir(self.root)
        self.csv_files = [file for file in self.csv_list if
                          file.endswith(".csv") and 'filtering' not in file and 'time_wrap' not in file]

        self.csv_file_paths = [os.path.join(self.root, csv_file) for csv_file in self.csv_files]
        print(f"{split} counts:{len(self.csv_file_paths)}")
        self.augmentation = augmentation

        self.padding_length = padding_length
        self.split = split
        self.augmentation_funcs = [add_noise, time_warp, signal_distortion, filtering]
        self.args = args  # Store the args parameter

        self.class_counts = self.initialize_class_counts()
        print(self.class_counts)

    def initialize_class_counts(self):
        class_counts = {}
        for csv_path in self.csv_file_paths:
            try:
                # Extract label as before
                file_name = csv_path.split('/')[-1]
                label_str = file_name.split('_')[-1].replace('.csv', '')
                label = int(label_str) - 1

                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1

            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
        return class_counts

    def __getitem__(self, index):
        if index < 0 or index >= len(self.csv_file_paths):
            return None
        csv_path = self.csv_file_paths[index]

        try:
            data = pd.read_csv(csv_path, delimiter=',')

            if self.args.dataset_name == 'task2_v2_x8_2':
                heog = data['Dev1/ai3'].values
                veog = data['Dev1/ai0'].values
            else:
                # 1열이 h, 2열이 v
                heog = data['Dev1.ai3'].values
                veog = data['Dev1.ai0'].values

            # # 4. 필터링 (Moving Average)
            eog_data = np.vstack((heog, veog)).T


            # # 5. 미분 계산 및 스케일링
            data = torch.tensor(eog_data, dtype=torch.float32)

            # 6. 앞뒤 데이터 자르기
            data = trim_eog_data(data, trim_length=50)  # 여기에 trim 함수 적용

            trim_data_np = data.numpy()

            multigap_H = sliding_multi(trim_data_np[:, 0], 30, 0, 0.5)
            multigap_V = sliding_multi(trim_data_np[:, 1], 30, 0, 0.5)

            multigap_H_tensor = torch.tensor(multigap_H, dtype=torch.float32)
            multigap_V_tensor = torch.tensor(multigap_V, dtype=torch.float32)

            diff_data = torch.stack([multigap_H_tensor, multigap_V_tensor], dim=-1)

            # diff_data와 크기 맞추기
            data_trimmed_for_stack = data[:diff_data.shape[0], :]

            # # 🔹 최종 데이터 합치기 → (seq_len, 4)
            data = torch.stack(
                [data_trimmed_for_stack[:, 0], data_trimmed_for_stack[:, 1], diff_data[:, 0], diff_data[:, 1]], dim=-1
            )

            # 7. 패딩 적용
            if data.size(0) < self.padding_length:
                # 패딩이 필요한 경우
                padding = torch.zeros((self.padding_length - data.size(0), data.size(1)))
                data = torch.cat((data, padding), dim=0)
            elif data.size(0) > self.padding_length:
                # 데이터가 패딩 길이보다 긴 경우, 자르기
                data = data[:self.padding_length, :]

            label = int(csv_path.split('/')[-1].split('_')[-1].replace('.csv', '')) - 1
            # print(label)
            ###########################ORG############################################
            self.class_counts[label] += 1
            if self.augmentation:
                for func in self.augmentation_funcs:
                    eog_data = func(eog_data)

            # 파일 이름 분석하여 레이블 추출
            if sys.platform == 'win32' or sys.platform == 'win64':
                file_name = csv_path.split('\\')[-1]  # 파일 경로에서 파일 이름만 추출
            else:
                file_name = csv_path.split('/')[-1]  # 파일 경로에서 파일 이름만 추출

            if self.split == 'train_aug' or self.split == 'train_aug2':
                label_str = file_name.split('_')[5]
            else:
                label_str = file_name.split('_')[-1].replace('.csv', '')

            label = int(label_str)  # 문자열 레이블을 정수로 변환
            label = label - 1  # 1, 2, 3, 4로 라벨링 되어있음
            label_tensor = torch.tensor(label, dtype=torch.long) # 라벨을 torch.tensor로 변환

            start_x = float(file_name.split('_')[1])
            start_y = float(file_name.split('_')[2])

            start_x = torch.tensor([start_x], dtype=torch.float32)  # 스칼라 값을 포함하는 텐서로 변환
            start_y = torch.tensor([start_y], dtype=torch.float32)  # 스칼라 값을 포함하는 텐서로 변환
            start = torch.tensor([[start_x, start_y]],
                                 dtype=torch.float32)  # 두 개의 스칼라 값을 포함하는 텐서로 변환(label_tensor, csv_path)

            return start, data, label_tensor, csv_path
        except Exception as e:
            print(f"Error at {csv_path}: {e}")
            return None

    def __len__(self):
        return len(self.csv_list)

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]

        if not batch or len(batch) == 0:
            pass
        else:
            start, data, label, csv_path = zip(*batch)
            start = torch.stack(start, dim=0)
            # label = torch.stack(label, dim=0).squeeze(1)
            label = torch.stack(label, dim=0)
            padded_data = torch.zeros(len(data), self.padding_length, data[0].shape[1])  # 데이터의 두 번째 차원 크기에 주의

            for i, d in enumerate(data):
                padded_data[i, :min(d.shape[0], self.padding_length), :] = d[:self.padding_length, :]

            return start, padded_data, label, csv_path


class EOGTestDataset(Dataset):
    def __init__(self, root='./EOG-Data', split='test', padding_length=751, augmentation=False, args=None):
        if split is not None:
            self.root = os.path.join(root, split)
        else:
            self.root = root

        self.csv_list = os.listdir(self.root)
        self.csv_files = [file for file in self.csv_list if file.endswith(".csv")]
        # print(self.csv_files)
        self.csv_file_paths = [os.path.join(self.root, csv_file) for csv_file in self.csv_files]
        self.padding_length = padding_length
        self.args = args
        print(f"{split} counts:{len(self.csv_file_paths)}")

    def __getitem__(self, index):
        if index < 0 or index >= len(self.csv_file_paths):
            return None
        csv_path = self.csv_file_paths[index]

        try:
            data = pd.read_csv(csv_path, delimiter=',')
            # print(data)
            if self.args.dataset_name == 'task2_v2_x8_2' or self.args.dataset_name == 'task2_v2_x8_iot':
                heog = data['Dev1/ai3'].values
                veog = data['Dev1/ai0'].values
            else:
                # 1열이 h, 2열이 v
                heog = data['Dev1.ai3'].values
                veog = data['Dev1.ai0'].values

            # # 4. 필터링 (Moving Average)
            eog_data = np.vstack((heog, veog)).T

            # # 5. 미분 계산 및 스케일링
            data = torch.tensor(eog_data, dtype=torch.float32)

            # 6. 앞뒤 데이터 자르기
            data = trim_eog_data(data, trim_length=50)  # 여기에 trim 함수 적용

            trim_data_np = data.numpy()

            multigap_H = sliding_multi(trim_data_np[:, 0], 30, 0, 0.5)
            multigap_V = sliding_multi(trim_data_np[:, 1], 30, 0, 0.5)

            multigap_H_tensor = torch.tensor(multigap_H, dtype=torch.float32)
            multigap_V_tensor = torch.tensor(multigap_V, dtype=torch.float32)
            #
            diff_data = torch.stack([multigap_H_tensor, multigap_V_tensor], dim=-1)

            # # diff_data와 크기 맞추기
            data_trimmed_for_stack = data[:diff_data.shape[0], :]

            # # 🔹 최종 데이터 합치기 → (seq_len, 4)
            data = torch.stack(
                [data_trimmed_for_stack[:, 0], data_trimmed_for_stack[:, 1], diff_data[:, 0], diff_data[:, 1]], dim=-1
            )

            # 패딩 적용
            if data.size(0) < self.padding_length:
                # 패딩이 필요한 경우
                padding = torch.zeros((self.padding_length - data.size(0), data.size(1)))
                data = torch.cat((data, padding), dim=0)
            elif data.size(0) > self.padding_length:
                # 데이터가 패딩 길이보다 긴 경우, 자르기
                data = data[:self.padding_length, :]

            # 데이터가 유효하지 않을 경우 처리
            if data is None:
                raise ValueError(f"Invalid data at index {index}")

            if sys.platform == 'win32' or sys.platform == 'win64':
                file_name = csv_path.split('\\')[-1]  # 파일 경로에서 파일 이름만 추출
            else:
                file_name = csv_path.split('/')[-1]  # 파일 경로에서 파일 이름만 추출

            if self.args.dataset_name == 'task2_v2_x8_iot':
                label_str = file_name.split('_')[2]
                start_x = 0.0
                start_y = 0.0
            else:
                label_str = file_name.split('_')[-1].replace('.csv', '')
                start_x = float(file_name.split('_')[1])
                start_y = float(file_name.split('_')[2])

            label = int(label_str)  # 문자열 레이블을 정수로 변환
            label = label - 1  # 1, 2, 3, 4로 라벨링 되어있음
            label_tensor = torch.tensor(label, dtype=torch.long)

            start_x = torch.tensor([start_x], dtype=torch.float32)  # 스칼라 값을 포함하는 텐서로 변환
            start_y = torch.tensor([start_y], dtype=torch.float32)  # 스칼라 값을 포함하는 텐서로 변환
            start = torch.tensor([[start_x, start_y]], dtype=torch.float32)  # 두 개의 스칼라 값을 포함하는 텐서로 변환
            return start, data, label_tensor, csv_path
        except Exception as e:
            print(f"Error at {csv_path}: {e}")
            return None

    def __len__(self):
        return len(self.csv_list)

    def collate_fn(self, batch):
        # original_len = len(batch)
        batch = [item for item in batch if item is not None]

        if not batch or len(batch) == 0:
            pass
        else:
            start, data, label, csv_path = zip(*batch)
            start = torch.stack(start, dim=0)
            label = torch.stack(label, dim=0)
            padded_data = torch.zeros(len(data), self.padding_length, data[0].shape[1])  # 데이터의 두 번째 차원 크기에 주의

            for i, d in enumerate(data):
                padded_data[i, :min(d.shape[0], self.padding_length), :] = d[:self.padding_length, :]

            return start, padded_data, label, csv_path


if __name__ == '__main__':
    dataset = EOGTestDataset(root='data/crossval_dataset1/augmented_dataset', split='train')

    start, data, label, csv_path = dataset[0]

    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    for batch in dataloader:
        # 이곳에서 배치 처리된 데이터를 사용할 수 있습니다.
        start, data, label, csv_path = batch
        # 여기서 필요한 작업을 수행하세요.
