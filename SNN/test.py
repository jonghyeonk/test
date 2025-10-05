import os
import numpy as np
import pandas as pd
import random
import argparse
from ast import literal_eval

import torch
import torch.backends.cudnn as cudnn

from snn_model import SNN


def set_seed():
    """ 모든 랜덤 시드를 설정하는 함수 """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)


def calculate_accuracy(model, X, Y):
    model.eval()
    with torch.no_grad():
        spk_rec, _ = model(X.view(len(X), -1), num_steps)
        _, pred = spk_rec.sum(dim=0).max(1)
        correct = (pred == Y).sum().item()
        accuracy = correct / len(Y) * 100
    return accuracy, pred


def test(df, X, Y, model):
    test_mask1 = df.split == 'test'

    testX, testY = X[test_mask1], Y[test_mask1]

    testX = testX.to(DEVICE)
    testY = testY.to(DEVICE)

    test_accuracy1, pred_test1 = calculate_accuracy(model, testX, testY)

    # Convert tensors to numpy arrays
    testX_np = testX.cpu().numpy()
    testY_np = testY.cpu().numpy()
    pred_test_np = pred_test1.cpu().numpy()

    # The shape of testX1 is (N, L, C). We need to flatten it for CSV.
    n_samples, seq_len, n_channels = testX_np.shape

    # Repeat labels and predictions to match the flattened X data
    repeated_testY = np.repeat(testY_np, seq_len)
    repeated_testY += 1
    repeated_pred_test = np.repeat(pred_test_np, seq_len)
    repeated_pred_test += 1

    # Prepare data for DataFrame
    data_to_save = {
        'true_label': repeated_testY,
        'predicted_label': repeated_pred_test
    }

    # Add the input features to the dictionary
    for i in range(n_channels):
        data_to_save[f'X_channel_{i}'] = testX_np[:, :, i].flatten()

    # Create and save the DataFrame
    df_results = pd.DataFrame(data_to_save)

    df_results.to_csv(args.output, index=False)

    print(f"Test results saved to {args.output}")
    print(f"Test: {test_accuracy1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SNN Train')
    parser.add_argument('--input', type=str, default='./input_result.csv', help='input data name (e.g., input_x8.csv)')
    parser.add_argument('--output', type=str, default='SNN Result/input_result.csv', help='output data name (e.g., result_x8.csv)')
    parser.add_argument('--model_path', type=str, default='./SNN ckpt/snn_model_x8.pth', help='Path to the trained SNN model')

    parser.add_argument('--num_steps', type=int, default=15, help='SNN step size for testing')
    parser.add_argument('--num_hidden', type=int, default=15, help='Number of SNN hidden layer')
    parser.add_argument('--num_classes', type=int, default=8, help='4, 8')
    parser.add_argument('--beta', type=float, default=0.95, help='SNN Leaky Beta value')
    parser.add_argument('--thres', type=float, default=0.8, help='SNN Leaky threshold value')

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the inference on')

    args = parser.parse_args()

    set_seed()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    DEVICE = args.device
    print(DEVICE)

    # 하이퍼파라미터 설정
    beta = args.beta
    thres = args.thres

    num_steps = args.num_steps
    num_hidden = args.num_hidden

    input_file_path = args.input

    df = pd.read_csv(input_file_path, converters={'base_H': literal_eval, 'base_V': literal_eval})
    X = torch.tensor(df[['gap_H', 'gap_V']].apply(lambda x: [x.tolist()], axis=1).tolist(), dtype=torch.float)
    Y = torch.tensor(df['class'].tolist(), dtype=torch.int64)

    num_inputs = X.shape[2]
    num_outputs = args.num_classes

    state_dict = torch.load(args.model_path)
    modelX = SNN(num_inputs, num_hidden, num_outputs, beta, thres).to(DEVICE)
    modelX.load_state_dict(state_dict=state_dict)

    test(df, X, Y, modelX)
