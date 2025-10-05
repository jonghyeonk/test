import torch
import torch.nn as nn
import tqdm
import os
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from collections import defaultdict

# 매 에폭마다 누적된 잘못된 예측을 추적하는 딕셔너리
persistent_incorrect_cases = defaultdict(int)

@torch.no_grad()
def evaluate(args, model, testset_loader, device, criterion, criterion_name="Loss", epoch=-1, num_incorrect_to_remove=0):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_predictions = []
    incorrect_cases = []
    correct_cases = []

    # args.num_classes에 기반한 전체 클래스 범위 생성
    labels = list(range(args.num_classes))

    # 모든 배치를 처리하며 잘못된 예측과 올바른 예측을 구분합니다.
    for i, batch in enumerate(tqdm.tqdm(testset_loader)):
        if batch is None:
            continue
        start, eog_data, label, csv_path = batch
        start = start.to(device)
        label = label.to(device)
        eog_data = eog_data.to(device)

        output, _ = model(eog_data)

        loss = criterion(output, label)
        total_loss += loss.item()

        # softmax 적용하여 확률 계산, 가장 높은 확률을 가진 인덱스를 예측으로 사용
        probabilities = F.softmax(output, dim=1)
        predictions = probabilities.argmax(dim=1)

        for j in range(label.size(0)):
            if predictions[j] != label[j]:
                incorrect_cases.append((predictions[j].item(), label[j].item(), csv_path[j], loss.item()))
                persistent_incorrect_cases[csv_path[j]] += 1
            else:
                correct_cases.append((predictions[j].item(), label[j].item(), csv_path[j], loss.item()))

        # 실제 레이블과 예측이 일치하는 경우의 수를 세어 정확도 계산
        correct_predictions += (predictions == label).sum().item()
        total_samples += label.size(0)

    if total_samples > 0:
        # 제거 전 성능 계산
        initial_outputs = [case[0] for case in incorrect_cases + correct_cases]
        initial_labels = [case[1] for case in incorrect_cases + correct_cases]

        initial_outputs = torch.tensor(initial_outputs)
        initial_labels = torch.tensor(initial_labels)

        initial_precision, initial_recall, initial_f1, _ = precision_recall_fscore_support(initial_labels, initial_outputs, average='weighted', zero_division=0)
        initial_cm = confusion_matrix(initial_labels, initial_outputs, labels=labels)  # labels 추가

        initial_accuracy = (initial_outputs == initial_labels).sum().item() / len(initial_labels) * 100
        initial_avg_loss = total_loss / len(testset_loader)

        if num_incorrect_to_remove > 0 and args.incorrect_eval_show_mode == True and epoch % 50 == 0:
            print(f'\n[Epoch {epoch}] Before Removing Incorrect Predictions:')
            print(f'Loss: {initial_avg_loss:.4f}, Accuracy: {initial_accuracy:.2f}%, Precision: {initial_precision:.4f}, Recall: {initial_recall:.4f}, F1: {initial_f1:.4f}')
            print(f'Confusion Matrix:\n{initial_cm}')
        else:
            print(f'Evaluation - Loss: {initial_avg_loss:.4f}, Accuracy: {initial_accuracy:.2f}%, Precision: {initial_precision:.4f}, Recall: {initial_recall:.4f}, F1: {initial_f1:.4f}')

        if num_incorrect_to_remove > 0  and args.incorrect_eval_show_mode == True:
            # 잘못된 예측을 특정 개수만큼 제거
            incorrect_cases.sort(key=lambda x: x[3], reverse=True)
            removed_cases = incorrect_cases[:num_incorrect_to_remove]
            remaining_incorrect_cases = incorrect_cases[num_incorrect_to_remove:]

            # 제거 후 성능 계산
            remaining_cases = remaining_incorrect_cases + correct_cases
            remaining_outputs = [case[0] for case in remaining_cases]
            remaining_labels = [case[1] for case in remaining_cases]

            remaining_outputs = torch.tensor(remaining_outputs)
            remaining_labels = torch.tensor(remaining_labels)

            final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(remaining_labels, remaining_outputs, average='weighted', zero_division=0)
            final_cm = confusion_matrix(remaining_labels, remaining_outputs, labels=labels)  # labels 추가

            final_accuracy = (remaining_outputs == remaining_labels).sum().item() / len(remaining_labels) * 100
            final_avg_loss = total_loss / len(testset_loader)  # final_avg_loss를 올바르게 정의

            print(f'\n[Epoch {epoch}] After Removing Incorrect Predictions:')
            print(f'Final Loss: {initial_avg_loss:.4f}, Final Accuracy: {final_accuracy:.2f}%, Final Precision: {final_precision:.4f}, Final Recall: {final_recall:.4f}, Final F1: {final_f1:.4f}')
            print(f'Final Confusion Matrix:\n{final_cm}')

            print(f'\n[Epoch {epoch}] Removed Incorrect Predictions:')
            for predicted_label, true_label, path, _ in removed_cases:
                print(f'Path: {path}, True Label: {true_label}, Predicted Label: {predicted_label}')
        else:
            final_avg_loss = initial_avg_loss
            final_accuracy = initial_accuracy
            final_precision = initial_precision
            final_recall = initial_recall
            final_f1 = initial_f1
            final_cm = initial_cm
            removed_cases = []
        # 에폭 종료 시, 누적된 잘못된 예측을 기반으로 최종 성능 평가
        if num_incorrect_to_remove > 0 and epoch % 50 == 0 and args.incorrect_eval_show_mode == True:
            persistent_incorrect_list = sorted(persistent_incorrect_cases.items(), key=lambda x: x[1], reverse=True)
            persistent_incorrect_paths = [item[0] for item in persistent_incorrect_list[:num_incorrect_to_remove]]

            # 누적된 잘못된 예측 경로를 제거하고 최종 성능을 평가
            final_remaining_cases = [case for case in remaining_cases if case[2] not in persistent_incorrect_paths]

            final_remaining_outputs = torch.tensor([case[0] for case in final_remaining_cases])
            final_remaining_labels = torch.tensor([case[1] for case in final_remaining_cases])

            final_final_precision, final_final_recall, final_final_f1, _ = precision_recall_fscore_support(final_remaining_labels, final_remaining_outputs, average='weighted', zero_division=0)
            final_final_cm = confusion_matrix(final_remaining_labels, final_remaining_outputs, labels=labels)  # labels 추가

            final_final_accuracy = (final_remaining_outputs == final_remaining_labels).sum().item() / len(final_remaining_labels) * 100

            print(f'\n[Epoch {epoch}] After Removing Most Persistent Incorrect Predictions:')
            print(f'Test Final Average Accuracy: {final_final_accuracy:.2f}%')
            print(f'Final Precision: {final_final_precision:.4f}, Final Recall: {final_final_recall:.4f}, Final F1 Score: {final_final_f1:.4f}')
            print(f'Final Confusion Matrix:\n{final_final_cm}')
    else:
        print("No samples to evaluate.")
        final_avg_loss, final_accuracy, final_precision, final_recall, final_f1, final_cm = None, None, None, None, None, None
        removed_cases = []

    return final_avg_loss, final_accuracy, final_precision, final_recall, final_f1, final_cm, removed_cases
