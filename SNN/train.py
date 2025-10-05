import os
import numpy as np
import pandas as pd
import argparse
import random
from ast import literal_eval

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn

import snntorch as snn
from snn_model import SNN


def set_seed(seed_value):
    # Step3: 교차 검증
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)


def train(df, X, Y):
    train_mask = df.split == 'train'
    val_mask = df.split == 'eval'
    test_mask = df.split == 'test'

    trainX, trainY = X[train_mask], Y[train_mask]
    valX, valY = X[val_mask], Y[val_mask]
    testX, testY = X[test_mask], Y[test_mask]

    trainX = trainX.to(DEVICE)
    trainY = trainY.to(DEVICE)
    valX = valX.to(DEVICE)
    valY = valY.to(DEVICE)
    testX = testX.to(DEVICE)
    testY = testY.to(DEVICE)

    modelX = SNN(num_inputs, num_hidden, num_outputs, beta, thres).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(modelX.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    train_dataset = TensorDataset(trainX, trainY)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 정확도 계산
    def calculate_accuracy(X, Y):
        modelX.eval()
        with torch.no_grad():
            spk_rec, _ = modelX(X.view(len(X), -1), num_steps)
            _, pred = spk_rec.sum(dim=0).max(1)
            correct = (pred == Y).sum().item()
            accuracy = correct / len(Y) * 100
        return accuracy

    # epoch 별 정확도 및 손실 값 저장
    train_accuracies = []
    eval_accuracies = []
    test_accuracies = []
    train_losses = []  # 여기에서 train_losses 리스트를 초기화합니다.

    # 학습
    for epoch in range(num_epochs):
        modelX.train()
        correct_train = 0
        total_train = 0
        epoch_loss = 0  # 에포크 손실값 초기화

        # 모델 훈련
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            spk_rec, mem_rec = modelX(inputs.view(batch_size, -1), num_steps)
            loss_val = torch.zeros((1), dtype=torch.float, device=DEVICE)

            for step in range(num_steps):
                loss_val += loss_function(spk_rec[step], targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            epoch_loss += loss_val.item()  # 에포크 손실값 누적
            _, predicted = spk_rec.sum(dim=0).max(1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_accuracy = 100 * correct_train / total_train
        eval_accuracy = calculate_accuracy(valX, valY)
        test_accuracy = calculate_accuracy(testX, testY)

        # 시각화 데이터 저장
        train_accuracies.append(train_accuracy)
        eval_accuracies.append(eval_accuracy)
        test_accuracies.append(test_accuracy)
        train_losses.append(epoch_loss)  # 누적된 에포크 손실값을 저장

        if ((epoch + 1) % div == 0) and (epoch > 0):  # Add this line
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                  f"Train: {train_accuracy:.2f}%, Eval: {eval_accuracy:.2f}%, Test: {test_accuracy:.2f}%")

    # --- Save the Trained Model ---
    print("\nTraining finished. Saving model...")
    model_path = args.output
    torch.save(modelX.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SNN Test')
    parser.add_argument('--input', type=str, default='./input_result.csv', help='input data name (e.g., input_x8.csv)')
    parser.add_argument('--output', type=str, default='./SNN ckpt/new_snn_model.pth', help='save model name (e.g., snn_model_x8.pth)')

    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--epochs', type=int, default=85, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')  # 1e-3 #2.5e-04

    parser.add_argument('--num_steps', type=int, default=15, help='SNN step size for testing')
    parser.add_argument('--num_hidden', type=int, default=15, help='Number of SNN hidden layer')
    parser.add_argument('--num_classes', type=int, default=8, help='4, 8')
    parser.add_argument('--beta', type=float, default=0.95, help='SNN Leaky Beta value')
    parser.add_argument('--thres', type=float, default=0.8, help='SNN Leaky threshold value')
    parser.add_argument('--print_freq', type=int, default=5, help='Training result print frequency')

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the inference on')

    args = parser.parse_args()

    seed_value = 47
    set_seed(seed_value)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    # 하이퍼파라미터 설정
    beta = args.beta
    thres = args.thres

    learning_rate = args.lr
    num_epochs = args.epochs
    num_steps = args.num_steps
    num_hidden = args.num_hidden
    num_outputs = args.num_classes
    batch_size = args.batch_size
    div = args.print_freq

    df = pd.read_csv(args.input, converters={'base_H': literal_eval, 'base_V': literal_eval,
                                                 'multigap_H': literal_eval, 'multigap_V': literal_eval})

    df['class'] = df['caseid'].apply(lambda x: x.split('_')[5])
    df['class'] = df['class'].apply(lambda x: int(x.split('.')[0]) - 1)

    X = torch.tensor(df[['gap_H', 'gap_V']].apply(lambda x: [x.tolist()], axis=1).tolist(), dtype=torch.float)
    Y = torch.tensor(df['class'].tolist(), dtype=torch.int64)

    num_inputs = X.shape[2]

    train(df, X, Y)