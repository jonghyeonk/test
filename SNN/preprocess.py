import pandas as pd
import numpy as np
from os import walk
import warnings
import argparse

import torch
import random
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LSTM.model import EOGLSTMAttention
from utils import EOGPreprocessDataset

warnings.filterwarnings(action='ignore')
import os


def sliding_multi(x, size=30, wait=150, alpha=0.5):
    gap_hist = []
    save = []
    gap_std = 0
    for i in range(wait, len(x) - size + 1 - 10):
        gap = 0
        window = x[i:(i + size - 1)]
        seemax = np.max(window)
        seemin = np.min(window)
        see = seemax - seemin
        save.append(see)

        if i > wait + size:
            gap_std = np.std(save[0:(len(save) - 1)])
            cond = ((see > save[len(save) - 2] + alpha * gap_std) and (gap_std > 0))
            if cond:
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

    return gap_hist, gap_std


def map_attention_to_raw(raw_eog, attn, trim_length=100, padding_length=751):
    """
    모델에서 나온 attention 값을 raw EOG 데이터 길이에 맞게 매핑합니다.

    Parameters:
    - raw_eog: (L_raw, 2) 원본 EOG 데이터
    - attn: (padding_length,) 모델 attention 값 (trim + padding 후 길이)
    - trim_length: 앞뒤로 잘라낸 길이
    - padding_length: 모델 입력 길이

    Returns:
    - attn_mapped: (L_raw,) 원본 EOG 길이에 맞춘 attention 값
    """
    L_raw = raw_eog.shape[0]  # 750
    valid_len = L_raw - 2 * trim_length  # 550

    attn_mapped = np.zeros(L_raw)

    # trim 이후 유효한 attention만 가져옴
    attn_valid = attn[:valid_len]  # (550,)

    # 원본 데이터의 [100:650] 구간에 매핑
    attn_mapped[trim_length:trim_length + valid_len] = attn_valid

    return attn_mapped


def extract_attention_window(signal, attn_weights, window_size=100, method="median"):
    """
    Attention 최고점 주변 데이터를 추출하고 중심화 보정 + gap 계산
    """
    signal = np.array(signal)
    attn_weights = np.array(attn_weights)

    peak_idx = int(np.argmax(attn_weights))
    start = max(0, peak_idx - window_size)
    end = min(len(signal), peak_idx + window_size)

    if start == 0:
        end = end - (peak_idx - window_size)
    elif end == len(signal):
        start = start - (peak_idx + window_size - len(signal))

    extracted = signal[start:end]

    return extracted.tolist()


def process_input_file(root, model, args, device, window_size=100, method="median"):
    input_df = pd.DataFrame()

    model.eval()
    padding_length = 751
    folder = ['train', 'test', 'eval']

    for loc in folder:
        dataset = EOGPreprocessDataset(root=root, split=loc, padding_length=padding_length, args=args)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=dataset.collate_fn, num_workers=args.num_workers,
                                 pin_memory=False, drop_last=False)

        all_caseid, all_x, all_y = [], [], []
        all_base_H, all_base_V, all_gap_H, all_gap_V = [], [], [], []
        all_multi_gap_H, all_multi_gap_V = [], []
        all_gap_stds_H, all_gap_stds_V = [], []
        all_classes = []

        for i, batch_data in enumerate(data_loader):
            start, eog_data, label, csv_path, org_data = batch_data
            eog_data = eog_data.to(device)

            output, (attn_veog, attn_heog) = model(eog_data)

            for b in range(eog_data.size(0)):
                name = os.path.basename(csv_path[b])
                x = float(name.split('_')[3])
                y = float(name.split('_')[4])
                cls = int(name.split('_')[5].split('.')[0]) - 1
                caseid = name

                # 중복 처리
                if caseid in all_caseid:
                    caseid = f"{caseid}+dup{all_caseid.count(caseid)}"

                ## Raw EOG Data
                raw_heog_signal = org_data[b][0]
                raw_veog_signal = org_data[b][1]

                attn_h = attn_heog[b].detach().cpu().numpy().squeeze()
                attn_v = attn_veog[b].detach().cpu().numpy().squeeze()

                # attention을 원래 raw 길이로 매핑
                attn_v_raw = map_attention_to_raw(raw_veog_signal, attn_v, trim_length=0, padding_length=751)
                attn_h_raw = map_attention_to_raw(raw_heog_signal, attn_h, trim_length=0, padding_length=751)

                base_H = extract_attention_window(raw_heog_signal, attn_h_raw, window_size, method)
                base_V = extract_attention_window(raw_veog_signal, attn_v_raw, window_size, method)

                multigap_H, gap_std_H = sliding_multi(base_H, 30, 0, 0.5)
                multigap_V, gap_std_V = sliding_multi(base_V, 30, 0, 0.5)

                gap_H = multigap_H[np.argmax([np.abs(x) for x in multigap_H])]
                gap_V = multigap_V[np.argmax([np.abs(x) for x in multigap_V])]

                if np.abs(gap_H) < 20 or gap_std_H < 10:
                    multigap_H, gap_std_H = sliding_multi(raw_heog_signal, 30, 0, 0.5)
                    gap_H = multigap_H[np.argmax([np.abs(x) for x in multigap_H])]

                if np.abs(gap_V) < 20 or gap_std_V < 10:
                    multigap_V, gap_std_V = sliding_multi(raw_veog_signal, 30, 0, 0.5)
                    gap_V = multigap_V[np.argmax([np.abs(x) for x in multigap_V])]

                all_caseid.append(caseid)
                all_x.append(x)
                all_y.append(y)
                all_base_H.append(base_H)
                all_base_V.append(base_V)
                all_gap_H.append(gap_H)
                all_gap_V.append(gap_V)
                all_multi_gap_H.append(multigap_H)
                all_multi_gap_V.append(multigap_V)
                all_gap_stds_H.append(gap_std_H)
                all_gap_stds_V.append(gap_std_V)
                all_classes.append(cls)

        total = pd.DataFrame({
            "caseid": all_caseid,
            "x": all_x,
            "y": all_y,
            "base_H": all_base_H,
            "base_V": all_base_V,
            "gap_H": all_gap_H,
            "gap_V": all_gap_V,
            "multigap_H": all_multi_gap_H,
            "multigap_V": all_multi_gap_V,
            "gap_std_H": all_gap_stds_H,
            "gap_std_V": all_gap_stds_V,
            'class': all_classes
        })
        total['user'] = total['caseid'].apply(lambda x: x.split('_')[0])
        total['split'] = loc

        input_df = pd.concat([input_df, total]).reset_index(drop=True)

        # 문자열로 변환 (리스트 저장 시 CSV 호환성 위해)
    input_df['base_H'] = input_df['base_H'].astype(str)
    input_df['base_V'] = input_df['base_V'].astype(str)

    input_df.to_csv(args.output, index=False)


## E-skin data
def assign_coordinates(value):
    if value == 0:
        return -250, 70
    elif value == 1:
        return -90, 160
    elif value == 2:
        return 90, 160
    elif value == 3:
        return 250, 70
    elif value == 4:
        return 250, -70
    elif value == 5:
        return 90, -160
    elif value == 6:
        return -90, -160
    elif value == 7:
        return -250, -70
    else:
        # Handle unexpected values or errors
        return None, None


def process_e_skin_file(root, model, args, device, window_size=100, method="median"):
    model.eval()
    padding_length = 751
    eskin_df = pd.DataFrame()

    available_splits = [s for s in ['train', 'eval', 'test'] if os.path.isdir(os.path.join(root, s))]
    if not available_splits:
        available_splits = [None]  # 폴더가 하나도 없으면 test로 처리

    for loc in available_splits:
        dataset = EOGPreprocessDataset(root=root, split=loc, padding_length=padding_length, args=args)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=dataset.collate_fn, num_workers=args.num_workers,
                                 pin_memory=False, drop_last=False)

        all_caseid, all_x, all_y = [], [], []
        all_base_H, all_base_V, all_gap_H, all_gap_V = [], [], [], []
        all_multi_gap_H, all_multi_gap_V = [], []
        all_gap_std_H, all_gap_std_V = [], []
        all_classes = []

        for i, batch_data in enumerate(data_loader):
            start, eog_data, label, csv_path, org_data = batch_data
            eog_data = eog_data.to(device)

            output, (attn_veog, attn_heog) = model(eog_data)

            for b in range(eog_data.size(0)):
                name = os.path.basename(csv_path[b])
                cls = int(name.split('_')[2]) - 1
                x, y = assign_coordinates(cls)

                # cls = label[b]
                caseid = name

                # 중복 처리
                if caseid in all_caseid:
                    caseid = f"{caseid}+dup{all_caseid.count(caseid)}"

                ## Raw EOG Data
                raw_heog_signal = org_data[b][0]
                raw_veog_signal = org_data[b][1]

                attn_h = attn_heog[b].detach().cpu().numpy().squeeze()
                attn_v = attn_veog[b].detach().cpu().numpy().squeeze()

                # attention을 원래 raw 길이로 매핑
                attn_v_raw = map_attention_to_raw(raw_veog_signal, attn_v, trim_length=0, padding_length=751)
                attn_h_raw = map_attention_to_raw(raw_heog_signal, attn_h, trim_length=0, padding_length=751)

                base_H = extract_attention_window(raw_heog_signal, attn_h_raw, window_size, method)
                base_V = extract_attention_window(raw_veog_signal, attn_v_raw, window_size, method)

                multigap_H, gap_std_H = sliding_multi(base_H, 30, 0, 0.5)
                multigap_V, gap_std_V = sliding_multi(base_V, 30, 0, 0.5)

                gap_H = multigap_H[np.argmax([np.abs(x) for x in multigap_H])]
                gap_V = multigap_V[np.argmax([np.abs(x) for x in multigap_V])]

                if np.abs(gap_H) < 20 or gap_std_H < 10:
                    multigap_H, gap_std_H = sliding_multi(raw_heog_signal, 30, 0, 0.5)
                    gap_H = multigap_H[np.argmax([np.abs(x) for x in multigap_H])]

                if np.abs(gap_V) < 20 or gap_std_V < 10:
                    multigap_V, gap_std_V = sliding_multi(raw_veog_signal, 30, 0, 0.5)
                    gap_V = multigap_V[np.argmax([np.abs(x) for x in multigap_V])]

                all_caseid.append(caseid)
                all_x.append(x)
                all_y.append(y)
                all_base_H.append(base_H)
                all_base_V.append(base_V)
                all_gap_H.append(gap_H)
                all_gap_V.append(gap_V)
                all_multi_gap_H.append(multigap_H)
                all_multi_gap_V.append(multigap_V)
                all_gap_std_H.append(gap_std_H)
                all_gap_std_V.append(gap_std_V)
                all_classes.append(cls)

        total = pd.DataFrame({
            "caseid": all_caseid,
            "x": all_x,
            "y": all_y,
            "base_H": all_base_H,
            "base_V": all_base_V,
            "gap_H": all_gap_H,
            "gap_V": all_gap_V,
            "multigap_H": all_multi_gap_H,
            "multigap_V": all_multi_gap_V,
            "gap_std_H": all_gap_std_H,
            "gat_std_V": all_gap_std_V,
            'class': all_classes
        })

        total['user'] = total['caseid'].apply(lambda x: x.split('_')[0])

        total['base_H'] = total['base_H'].astype(str)
        total['base_V'] = total['base_V'].astype(str)
        total['split'] = 'test'

        eskin_df = pd.concat([eskin_df, total]).reset_index(drop=True)

    eskin_df.to_csv(args.output, index=False)


def set_seed(seed_value):
    """ 모든 랜덤 시드를 설정하는 함수 """
    torch.manual_seed(seed_value)  # 파이토치를 위한 시드 설정
    torch.cuda.manual_seed(seed_value)  # CUDA를 위한 시드 설정
    torch.cuda.manual_seed_all(seed_value)  # 모든 CUDA 디바이스를 위한 시드 설정
    np.random.seed(seed_value)  # 넘파이를 위한 시드 설정
    random.seed(seed_value)  # 파이썬 내장 랜덤 라이브러리를 위한 시드 설정
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 환경변수 PYTHONHASHSEED 설정

    # 아래 두 줄은 CuDNN의 비결정론적 행위를 방지
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SNN preprocessing script')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')  # 20
    parser.add_argument('--dataset_root', type=str, default=r'../data_jeonju_0904/task2_v2_x8', help='Name of the dataset')
    parser.add_argument('--dataset_name', type=str, default='eog_input', help='Name of the dataset')
    parser.add_argument('--num_classes', type=int, default=8, help='4, 8, 16, 32')
    parser.add_argument('--ckpt', type=str,
                        default="EOGRNNAttention/125_test_model.pth",
                        help='checkpoints path')
    parser.add_argument('--output', type=str, default="input_result.csv", help='output csv file name')
    parser.add_argument('--input_dim', type=int, default=1, help='input dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='어텐션 헤드의 개수')
    parser.add_argument('--cnn_hdden_size', type=int, default=4, help='CNN 레이어의 히든 사이즈')
    parser.add_argument('--lstm_hdden_size', type=int, default=40, help='LSTM 레이어의 히든 사이즈')
    parser.add_argument('--num_layers_lstm', type=int, default=2, help='LSTM 레이어의 개수')
    parser.add_argument('--dropout_lstm', type=float, default=0.43, help='LSTM 레이어의 드롭아웃 비율')
    parser.add_argument('--window_size', type=int, default=150)

    args = parser.parse_args()

    # 시드 설정
    seed_value = 47
    set_seed(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    padding_length = 751

    state_dict = torch.load(args.ckpt)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    num_heads = args.num_heads
    cnn_hdden_size = args.cnn_hdden_size
    lstm_hdden_size = args.lstm_hdden_size
    num_layers_lstm = args.num_layers_lstm
    dropout_lstm = args.dropout_lstm

    model = EOGLSTMAttention(input_dim=args.input_dim, hidden_dim=lstm_hdden_size, num_layers=num_layers_lstm,
                             num_classes=args.num_classes,
                             num_heads=num_heads, dropout=dropout_lstm)

    model.load_state_dict(state_dict=new_state_dict)
    model.to(device)
    # input_data_root = os.path.abspath(args.dataset_name)
    input_data_root = args.dataset_root
    win_size = args.window_size
    if args.dataset_name == "eskin":
        process_e_skin_file(input_data_root, model, args, device, window_size=win_size, method='median')
    else:
        process_input_file(input_data_root, model, args, device, window_size=win_size, method='median')
