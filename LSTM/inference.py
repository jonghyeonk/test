import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict
import argparse
from dataset import EOGTestDataset  # 필요한 데이터셋 모듈
from torch.utils.data import DataLoader
# DataParallel로 저장된 모델의 경우 'module.' 접두사를 제거
from collections import OrderedDict
import os
import numpy as np
from model import EOGLSTMNoAttention, EOGLSTMAttention, EOGLSTMOneWay, EOG1DCNN


# 매 에폭마다 누적된 잘못된 예측을 추적하는 딕셔너리
persistent_incorrect_cases = defaultdict(int)

@torch.no_grad()
def inference(args, model, loader, device, criterion, num_incorrect_to_remove=0, epoch=0):
    model.eval()
    all_outputs = []
    all_labels = []
    incorrect_cases = []
    correct_cases = []
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []

    # 모든 데이터를 처리하면서 잘못된 예측과 올바른 예측을 구분
    for i, batch in enumerate(tqdm.tqdm(loader)):
        start, eog_data, label, csv_path = batch
        eog_data = eog_data.to(device)
        label = label.to(device)

        output, _ = model(eog_data)

        loss = criterion(output, label)
        total_loss += loss.item()

        probabilities = F.softmax(output, dim=1)
        predicted_classes = probabilities.argmax(dim=1)

        correct_predictions += (predicted_classes == label).sum().item()
        total_samples += label.size(0)

        for j in range(label.size(0)):
            if predicted_classes[j] != label[j]:
                incorrect_cases.append((predicted_classes[j].item(), label[j].item(), csv_path[j], loss.item()))
                # 매 에폭마다 해당 경로의 잘못된 예측을 누적
                persistent_incorrect_cases[csv_path[j]] += 1
            else:
                correct_cases.append((predicted_classes[j].item(), label[j].item(), csv_path[j], loss.item()))

    # 제거 전 성능 계산
    initial_outputs = [case[0] for case in incorrect_cases + correct_cases]
    initial_labels = [case[1] for case in incorrect_cases + correct_cases]

    initial_outputs = torch.tensor(initial_outputs)
    initial_labels = torch.tensor(initial_labels)

    initial_precision, initial_recall, initial_f1, _ = precision_recall_fscore_support(initial_labels, initial_outputs, average='weighted', zero_division=0)
    initial_cm = confusion_matrix(initial_labels, initial_outputs)
    # print("Initial Confusion Matrix shape:", initial_cm.shape)  # Confusion Matrix의 크기 확인
    initial_accuracy = (initial_outputs == initial_labels).sum().item() / len(initial_labels) * 100
    initial_average_loss = total_loss / len(loader)


    if num_incorrect_to_remove > 0:
        print(f'\n[Epoch {epoch}] Before Removing Incorrect Predictions:')
    else:
        print(f'\n[Epoch {epoch}] Inference Results:')
    print(f'Test Average Loss: {initial_average_loss:.4f}')
    print(f'Test Average Accuracy: {initial_accuracy:.2f}%')
    print(f'Precision: {initial_precision:.4f}, Recall: {initial_recall:.4f}, F1 Score: {initial_f1:.4f}')
    print(f'Confusion Matrix:\n{initial_cm}')

    remaining_outputs = initial_outputs
    remaining_labels = initial_labels

    if num_incorrect_to_remove > 0 and args.incorrect_test_show_mode == True:
        # 잘못된 예측을 특정 개수만큼 제거
        incorrect_cases.sort(key=lambda x: x[3], reverse=True)  # 손실이 큰 순서대로 정렬
        removed_cases = incorrect_cases[:num_incorrect_to_remove]
        remaining_incorrect_cases = incorrect_cases[num_incorrect_to_remove:]

        # 제거 후 남아있는 데이터로 다시 성능 지표 계산
        remaining_cases = remaining_incorrect_cases + correct_cases
        remaining_outputs = [case[0] for case in remaining_cases]
        remaining_labels = [case[1] for case in remaining_cases]

        remaining_outputs = torch.tensor(remaining_outputs)
        remaining_labels = torch.tensor(remaining_labels)

        final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(remaining_labels, remaining_outputs, average='weighted', zero_division=0)
        final_cm = confusion_matrix(remaining_labels, remaining_outputs)

        final_accuracy = (remaining_outputs == remaining_labels).sum().item() / len(remaining_labels) * 100

        print(f'\n[Epoch {epoch}] After Removing Incorrect Predictions:')
        print(f'Test Average Accuracy: {final_accuracy:.2f}%')
        print(f'Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1 Score: {final_f1:.4f}')
        print(f'Confusion Matrix:\n{final_cm}')

        print(f'\n[Epoch {epoch}] Removed Incorrect Predictions:')
        for predicted_label, true_label, path, _ in removed_cases:
            print(f'Path: {path}, True Label: {true_label}, Predicted Label: {predicted_label}')

        print(f'\n[Epoch {epoch}] Persistent Incorrect Predictions:')
        for path, count in persistent_incorrect_cases.items():
            print(f'Path: {path}, Incorrect Count: {count}')
    else:
        final_accuracy = initial_accuracy
        final_precision = initial_precision
        final_recall = initial_recall
        final_f1 = initial_f1
        final_cm = initial_cm
        removed_cases = []

    # 에폭 종료 시, 누적된 잘못된 예측 카운트를 기반으로 최종적으로 제거
    if num_incorrect_to_remove > 0 and epoch % args.initial_inference_interval == 0 and args.incorrect_test_show_mode == True:
        persistent_incorrect_list = sorted(persistent_incorrect_cases.items(), key=lambda x: x[1], reverse=True)
        persistent_incorrect_paths = [item[0] for item in persistent_incorrect_list[:num_incorrect_to_remove]]

        # 누적된 잘못된 예측 경로를 제거하고 최종 성능을 평가
        final_remaining_cases = [case for case in remaining_cases if case[2] not in persistent_incorrect_paths]

        final_remaining_outputs = torch.tensor([case[0] for case in final_remaining_cases])
        final_remaining_labels = torch.tensor([case[1] for case in final_remaining_cases])

        final_final_precision, final_final_recall, final_final_f1, _ = precision_recall_fscore_support(final_remaining_labels, final_remaining_outputs, average='weighted', zero_division=0)
        final_final_cm = confusion_matrix(final_remaining_labels, final_remaining_outputs)

        final_final_accuracy = (final_remaining_outputs == final_remaining_labels).sum().item() / len(final_remaining_labels) * 100

        print(f'\n[Epoch {epoch}] After Removing Most Persistent Incorrect Predictions:')
        print(f'Test Final Average Accuracy: {final_final_accuracy:.2f}%')
        print(f'Final Precision: {final_final_precision:.4f}, Final Recall: {final_final_recall:.4f}, Final F1 Score: {final_final_f1:.4f}')
        print(f'Final Confusion Matrix:\n{final_final_cm}')

    return remaining_outputs.numpy(), remaining_labels.numpy(), final_accuracy, final_precision, final_recall, final_f1, final_cm, removed_cases


if __name__ == "__main__":
    # argparse를 사용하여 인자를 받을 수 있도록 설정
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--dataset_base_directory', type=str, default='/datasets', help='Base directory for the dataset')
    parser.add_argument('--dataset_name', type=str, default='data_jeonju_0904/task2_v2_x8', help='Dataset name (e.g., task1_v2_x4_2)')
    parser.add_argument('--model_path', type=str, default='ckpts/EOGRNNAttention/125_test_model.pth', help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--num_incorrect_to_remove', type=int, default=0, help='Number of incorrect predictions to remove during inference')
    parser.add_argument('--model_name', type=str, default='EOGLSTMAttention',
                        choices=["EOG1DCNN", "EOGLSTMOneWay", "EOGLSTMNoAttention",
                                 'EOGLSTMAttention'],
                        help='Model to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the inference on')
    # 새로 추가된 인자
    parser.add_argument('--attn_dropout_start_epoch', type=int, default=None, help='Epoch to start attention dropout')
    parser.add_argument('--input_dim', type=int, default=1, help='input dimension')
    parser.add_argument('--num_classes', type=int, default=8, help='4, 8, 16, 32')
    parser.add_argument('--num_heads', type=int, default=4, help='어텐션 헤드의 개수')
    parser.add_argument('--cnn_hdden_size', type=int, default=4, help='CNN 레이어의 히든 사이즈')
    parser.add_argument('--lstm_hdden_size', type=int, default=40, help='LSTM 레이어의 히든 사이즈')
    parser.add_argument('--num_layers_lstm', type=int, default=2, help='LSTM 레이어의 개수')
    parser.add_argument('--dropout_lstm', type=float, default=0.43, help='LSTM 레이어의 드롭아웃 비율')

    args = parser.parse_args()

    # Model selection based on argument
    if args.model_name == 'EOG1DCNN':
        model = EOG1DCNN(input_dim=args.input_dim, hidden_dim=args.cnn_hdden_size, num_classes=args.num_classes)
    elif args.model_name == 'EOGLSTMOneWay':
        model = EOGLSTMOneWay(input_dim=args.input_dim, hidden_dim=args.lstm_hdden_size, num_layers=args.num_layers_lstm, num_classes=args.num_classes, dropout=args.dropout_lstm)
    elif args.model_name == 'EOGLSTMNoAttention':
        model = EOGLSTMNoAttention(input_dim=args.input_dim, hidden_dim=args.lstm_hdden_size, num_layers=args.num_layers_lstm, num_classes=args.num_classes,
                                           dropout=args.dropout_lstm)
    elif args.model_name == 'EOGLSTMAttention':
        model = EOGLSTMAttention(input_dim=args.input_dim, hidden_dim=args.lstm_hdden_size, num_layers=args.num_layers_lstm, num_classes=args.num_classes,
                                num_heads=args.num_heads, dropout=args.dropout_lstm)

    # 저장된 가중치를 로드하면서 'module.' 접두사 제거
    state_dict = torch.load(args.model_path, map_location=torch.device(args.device))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 'module.' 접두사 제거
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()

    # 데이터셋 준비
    dataset_root = os.path.join(args.dataset_base_directory, args.dataset_name)
    padding_length = 751  # 필요한 경우 조정

    testset = EOGTestDataset(root=dataset_root, split='test', padding_length=padding_length, args=args)
    testset_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 평가 함수 호출
    criterion = nn.CrossEntropyLoss()  # 손실 함수 (필요한 경우 변경)

    outputs, labels, final_accuracy, final_precision, final_recall, final_f1, confusion_matrix, removed_cases = inference(
        args, model, testset_loader, device, criterion, args.num_incorrect_to_remove
    )