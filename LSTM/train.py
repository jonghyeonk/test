import torch
import torch.nn as nn
import tqdm
import argparse
from evaluate import evaluate
from dataset import EOGTestDataset
from inference import inference
from dataset import EOGDataset  # 변경된 import
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR
import os
import numpy as np
from collections import defaultdict
import random
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from model import EOGLSTMNoAttention, EOGLSTMAttention, EOGLSTMOneWay, EOG1DCNN


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5, reduction='mean'):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert logits to probabilities (assuming inputs are logits)
        probs = torch.softmax(inputs, dim=1)

        # One-hot encode targets if not already
        true_1_hot = torch.nn.functional.one_hot(targets, num_classes=probs.size(1)).type_as(inputs)

        # True Positives, False Positives & False Negatives
        true_pos = torch.sum(true_1_hot * probs, dim=1)
        false_neg = torch.sum(true_1_hot * (1 - probs), dim=1)
        false_pos = torch.sum((1 - true_1_hot) * probs, dim=1)

        # Tversky index
        Tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)

        # Focal Tversky loss
        F_loss = (1 - Tversky) ** self.gamma

        # Reduction to scalar
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


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


def save_plots(train_losses, val_losses, train_accuracies, val_accuracies, lrs,
               filename_prefix="training_validation_combined_curve"):
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # Creates 3 rows and 1 column of subplots

    # 전체 그래프에 대한 기본 글씨 크기를 설정
    plt.rcParams.update({'font.size': 14})  # 기본 글씨 크기 설정 (옵션)

    # Loss plot
    axs[0].plot(epochs, train_losses, label='Training Loss', color='tab:red')
    axs[0].plot(epochs, val_losses, label='Validation Loss', color='tab:blue')
    axs[0].set_xlabel('Epochs', fontsize=16)  # x축 레이블 글씨 크기 설정
    axs[0].set_ylabel('Loss', fontsize=16)  # y축 레이블 글씨 크기 설정
    axs[0].tick_params(axis='x', labelsize=14)  # x축 틱 글씨 크기 설정
    axs[0].tick_params(axis='y', labelsize=14)  # y축 틱 글씨 크기 설정
    axs[0].legend(loc='upper left', fontsize=14)  # 범례 글씨 크기 설정

    # Accuracy plot
    axs[1].plot(epochs, train_accuracies, label='Training Accuracy', color='tab:purple')
    axs[1].plot(epochs, val_accuracies, label='Validation Accuracy', color='tab:orange')
    axs[1].set_xlabel('Epochs', fontsize=16)
    axs[1].set_ylabel('Accuracy', fontsize=16)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].legend(loc='upper left', fontsize=14)

    # Learning Rate plot
    axs[2].plot(epochs, lrs, label='Learning Rate', color='tab:green', linestyle='--')
    axs[2].set_xlabel('Epochs', fontsize=16)
    axs[2].set_ylabel('Learning Rate', fontsize=16)
    axs[2].set_yscale('log')
    axs[2].tick_params(axis='x', labelsize=14)
    axs[2].tick_params(axis='y', labelsize=14)
    axs[2].legend(loc='upper left', fontsize=14)

    fig.tight_layout()  # Adjusts the subplots to fit into the figure area.
    plt.suptitle('Training and Validation Curves with Learning Rate', fontsize=20)  # 전체 제목 글씨 크기 설정
    plt.subplots_adjust(top=0.93)  # Adjust top spacing to accommodate suptitle
    plt.savefig(f"{filename_prefix}.png")
    plt.close()


def plot_attention_on_data(eog_data, attention_weights, filename_prefix):
    # 파일을 저장할 디렉토리 생성
    directory = os.path.dirname(filename_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # attention_weights는 (attention_weights_veog, attention_weights_heog)로 전달됩니다.
    attention_veog, attention_heog = attention_weights  # VEOG와 HEOG 어텐션 가중치 분리
    time_steps = np.arange(eog_data.shape[0])  # EOG 데이터의 시간 스텝 정의

    print(f"Before attention_veog shape: {attention_veog.shape}, attention_heog.shape: {attention_heog.shape}")

    # attention_veog와 attention_heog가 1차원인지 확인하고, 그렇지 않으면 축소
    if attention_veog.ndim > 1:
        attention_veog = attention_veog.squeeze()  # 차원 축소
    if attention_heog.ndim > 1:
        attention_heog = attention_heog.squeeze()  # 차원 축소

    print(f"After attention_veog shape: {attention_veog.shape}, attention_heog.shape: {attention_heog.shape}")

    # VEOG 신호와 어텐션 가중치 시각화
    fig, ax1 = plt.subplots()

    # 글씨 크기 조정
    plt.rcParams.update({'font.size': 14})

    ax1.set_xlabel('Time Steps', fontsize=16)  # x축 레이블 글씨 크기
    ax1.set_ylabel('VEOG Signal', color='tab:red', fontsize=16)  # y축 레이블 글씨 크기
    ax1.plot(time_steps, eog_data[:, 0], color='tab:red', label='VEOG')  # VEOG 신호 시각화
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=12)  # y축 틱 글씨 크기
    ax1.tick_params(axis='x', labelsize=12)  # x축 틱 글씨 크기

    ax2 = ax1.twinx()
    ax2.set_ylabel('Attention Weight', color='tab:blue', fontsize=16)  # y축 레이블 글씨 크기
    ax2.fill_between(time_steps, attention_veog[:len(time_steps)], alpha=0.5, color='tab:blue', label='Attention VEOG')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)  # y축 틱 글씨 크기

    fig.tight_layout()
    plt.subplots_adjust(top=0.85)  # 상단 여백 조정
    plt.title('VEOG Signal with Attention Weight', fontsize=18)  # 제목 글씨 크기
    plt.savefig(f'{filename_prefix}_veog.png')
    plt.close()

    # HEOG 신호와 어텐션 가중치 시각화
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time Steps', fontsize=16)
    ax1.set_ylabel('HEOG Signal', color='tab:green', fontsize=16)
    ax1.plot(time_steps, eog_data[:, 1], color='tab:green', label='HEOG')
    ax1.tick_params(axis='y', labelcolor='tab:green', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Attention Weight', color='tab:purple', fontsize=16)
    ax2.fill_between(time_steps, attention_heog[:len(time_steps)], alpha=0.5, color='tab:purple',
                     label='Attention HEOG')
    ax2.tick_params(axis='y', labelcolor='tab:purple', labelsize=12)

    fig.tight_layout()
    plt.subplots_adjust(top=0.85)  # 상단 여백 조정
    plt.title('HEOG Signal with Attention Weight', fontsize=18)
    plt.savefig(f'{filename_prefix}_heog.png')
    plt.close()


persistent_incorrect_cases_train = defaultdict(int)


def train_one_epoch(args, model, optimizer, criterion, trainset_loader, lr_scheduler, scaler, accumulation_steps,
                    device,
                    epoch):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # 예측과 라벨을 저장할 리스트를 초기화합니다.
    all_labels = []
    all_predictions = []
    ### 2024-08-19
    incorrect_cases = []
    correct_cases = []

    progress_bar = tqdm.tqdm(trainset_loader)

    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    for i, batch_data in enumerate(progress_bar):  # frames -> data
        if batch_data is None:
            continue

        # 기울기 초기화
        optimizer.zero_grad()

        start, eog_data, label, csv_path = batch_data
        start, label, eog_data = start.to(device), label.to(device), eog_data.to(device)
        model.to(device)

        iteration = epoch * len(trainset_loader) + i

        with autocast(enabled=False):
            output, attention_weights = model(eog_data)  # 두 개의 attention 가중치 받기

        loss = criterion(output, label)
        scaler.scale(loss).backward()

        if (iteration + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item()

        #############################################################
        # 어텐션 가중치 시각화: 첫 번째 배치 또는 선택된 배치에서만 수행
        if i == 0 and epoch % 10 == 0:  # 여기서는 첫 번째 배치를 예로 들고 있습니다.
            eog_data_cpu = eog_data[0].cpu().detach().numpy()  # 첫 번째 샘플의 EOG 데이터

            # attention_weights가 tuple인지 확인하여 처리
            if isinstance(attention_weights, tuple) and attention_weights is not None:
                # Tuple인 경우, VEOG와 HEOG 어텐션 가중치를 각각 처리

                attn_weight_veog, attn_weight_heog = attention_weights  # 첫 번째 레이어의 어텐션 가중치 추출
                if attn_weight_veog is not None and attn_weight_heog is not None:
                    # print(attn_weight_veog.shape, attn_weight_heog.shape)
                    attention_weights_veog_cpu = attn_weight_veog[0].cpu().detach().numpy()
                    attention_weights_heog_cpu = attn_weight_heog[0].cpu().detach().numpy()
                    attention_weights_cpu = (attention_weights_veog_cpu, attention_weights_heog_cpu)
                else:
                    attention_weights_cpu = (None, None)

            else:
                attention_weights_cpu = attention_weights[0].cpu().detach().numpy()

            if args.model_name == 'EOGLSTMAttention':
                # 각 배치에 대해 고유한 파일명을 사용하도록 수정합니다.
                filename_prefix_batch = f"/datasets/{args.output_dir}/attention_epoch_{epoch}_batch_{i}/"
                plot_attention_on_data(
                    eog_data=eog_data_cpu,
                    attention_weights=attention_weights_cpu,
                    filename_prefix=filename_prefix_batch
                )
            #############################################################

        # 확률 얻기 위해서 softmax 적용
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)

        # 예측과 실제 라벨을 저장합니다.
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total_predictions += label.size(0)
        correct_predictions += (predicted == label).sum().item()

        current_lr = optimizer.param_groups[0]['lr']

        ################ 0819 추가
        # 잘못된 예측 추적
        for j in range(label.size(0)):
            if predicted[j] != label[j]:
                incorrect_cases.append((predicted[j].item(), label[j].item(), csv_path[j], loss.item()))
                persistent_incorrect_cases_train[csv_path[j]] += 1
            else:
                correct_cases.append((predicted[j].item(), label[j].item(), csv_path[j], loss.item()))

        # 혼동 행렬, 정밀도, 재현율, F1 점수를 계산합니다.
        accuracy = 100.0 * correct_predictions / total_predictions

        if args.num_incorrect_to_remove > 0:
            # 잘못된 예측 제거
            incorrect_cases.sort(key=lambda x: x[3], reverse=True)
            removed_cases = incorrect_cases[:args.num_incorrect_to_remove]
            remaining_incorrect_cases = incorrect_cases[args.num_incorrect_to_remove:]

            remaining_cases = remaining_incorrect_cases + correct_cases
            remaining_outputs = [case[0] for case in remaining_cases]
            remaining_labels = [case[1] for case in remaining_cases]

            remaining_outputs = torch.tensor(remaining_outputs)
            remaining_labels = torch.tensor(remaining_labels)

            precision, recall, f1, _ = precision_recall_fscore_support(remaining_labels, remaining_outputs,
                                                                       average='weighted', zero_division=0)

            # 특정 에폭마다 제거 전후 성능 비교를 출력
            if epoch % 100 == 0:
                print(f'\n[Epoch {epoch}] After Removing Incorrect Predictions:')
                print(f'Accuracy: {accuracy:.2f}%')
                print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

                print(f'\n[Epoch {epoch}] Removed Incorrect Predictions:')
                for predicted_label, true_label, path, _ in removed_cases:
                    print(f'Path: {path}, True Label: {true_label}, Predicted Label: {predicted_label}')

                print(f'\n[Epoch {epoch}] Persistent Incorrect Predictions:')
                for path, count in persistent_incorrect_cases_train.items():
                    print(f'Path: {path}, Incorrect Count: {count}')

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted',
                                                                   zero_division=0)

        progress_bar.set_description(f"Epoch {epoch}")
        progress_bar.set_postfix({
            'Loss': f'{running_loss / (i + 1):.4f}',
            'Accuracy': f"{accuracy:.2f}%",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1': f"{f1:.4f}",
            'LR': f"{current_lr:.6e}",
        })

    return running_loss / len(trainset_loader), accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulation steps')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of workers')  # 20
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2.5e-04, help='Learning rate')  # 1e-3 #2.5e-04
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup duration')
    parser.add_argument('--input_dim', type=int, default=1, help='input dimension')

    parser.add_argument('--initial_validation_interval', type=int, default=1,
                        help='Initial validation interval')  # scheduler때문에
    parser.add_argument('--initial_inference_interval', type=int, default=25, help='Initial inference interval')

    parser.add_argument('--dataset_base_directory', type=str, default='/datasets', help='Base directory for data')

    parser.add_argument('--dataset_name', type=str, default='data_jeonju_0904/task2_v2_x8', help='Name of the dataset')
    parser.add_argument('--num_classes', type=int, default=8, help='4, 8, 16, 32')

    parser.add_argument('--output_dir', type=str,
                        default='result/case4_EOGLSTMAttention',
                        help='Directory to save model weights and outputs')
    parser.add_argument('--num_incorrect_to_remove', type=int, default=0,
                        help='Number of incorrect predictions to remove during inference')  # New Argument
    parser.add_argument('--incorrect_eval_show_mode', type=bool, default=True,
                        help='Enable or disable eval show mode (True or False)')
    parser.add_argument('--incorrect_test_show_mode', type=bool, default=True,
                        help='Enable or disable test show mode (True or False)')

    parser.add_argument('--model_name', type=str, default='EOGLSTMAttention',
                        choices=["EOG1DCNN", "EOGLSTMOneWay", "EOGLSTMNoAttention",
                                 'EOGLSTMAttention'],
                        help='Model to use')
    parser.add_argument('--attn_dropout_start_epoch', type=int, default=None,
                        help='Initial epoch to start dropout of attn layer')
    parser.add_argument('--weight_decay', type=float, default=7e-1, help='Weight decay for optimizer')
    parser.add_argument('--num_heads', type=int, default=4, help='어텐션 헤드의 개수')
    parser.add_argument('--cnn_hdden_size', type=int, default=4, help='CNN 레이어의 히든 사이즈')
    parser.add_argument('--lstm_hdden_size', type=int, default=40, help='LSTM 레이어의 히든 사이즈')
    parser.add_argument('--num_layers_lstm', type=int, default=2, help='LSTM 레이어의 개수')
    parser.add_argument('--dropout_lstm', type=float, default=0.43, help='LSTM 레이어의 드롭아웃 비율')

    args = parser.parse_args()
    print(args)

    # 시드 설정
    seed_value = 47
    set_seed(seed_value)

    # 출력 디렉토리 생성
    ab_output_dir = os.path.join(args.dataset_base_directory, args.output_dir)
    if not os.path.exists(ab_output_dir):
        os.makedirs(ab_output_dir, exist_ok=True)

    # 설정값을 파일로 저장
    config_file_path = os.path.join(ab_output_dir, 'training_config.txt')
    print(config_file_path)
    # 파일 저장 시도
    try:
        print(f"Attempting to save training configuration to: {config_file_path}")
        with open(config_file_path, 'w') as f:
            f.write("훈련 설정값:\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        print(f"Training configuration successfully saved to {config_file_path}")
    except Exception as e:
        print(f"Failed to save training configuration: {e}")

    # 결과 파일 경로 설정
    train_results_file = os.path.join(ab_output_dir, "train_results.txt")
    val_results_file = os.path.join(ab_output_dir, "val_results.txt")
    test_results_file = os.path.join(ab_output_dir, "test_results.txt")
    if not os.path.exists(ab_output_dir):
        os.makedirs(ab_output_dir)

    # Train, Val 결과 파일 초기화
    with open(train_results_file, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1\n")

    with open(val_results_file, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1\n")

    with open(test_results_file, "w") as f:
        f.write("Epoch\tAccuracy\tPrecision\tRecall\tF1\n")

    output_file_path = os.path.join(ab_output_dir, 'outputs.txt')
    best_weight_file_path = os.path.join(ab_output_dir, 'best_model.pth')
    last_weight_file_path = os.path.join(ab_output_dir, 'last_model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # padding_length 데이터셋 별 패딩 다르게
    # padding_length = 1500 if 'combine' or '3' in args.dataset_name else 500 if 'eog_coord_dataset2' in args.dataset_name else 1500
    # padding_length = 751 if 'eog_point_dataset_v1' or 'eog_point_dataset_Final' or 'eog_point_dataset_Final2' or 'crossval_dataset1' in args.dataset_name else 1500
    padding_length = 751
    root_directory = f"{args.dataset_base_directory}/{args.dataset_name}"
    # train 데이터셋 및 로더
    data_root = '/datasets/' + args.dataset_name
    print(f"data_root: {data_root}")
    trainset = EOGDataset(root=data_root, split='train', padding_length=padding_length,
                          augmentation=True, args=args)
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  collate_fn=trainset.collate_fn, num_workers=args.num_workers,
                                                  pin_memory=True, drop_last=False)

    # eval 데이터셋 및 로더
    evalset = EOGDataset(root=data_root, split='eval', padding_length=padding_length,
                         augmentation=False, args=args)  # 일반적으로 평가시에는 augmentation을 사용하지 않습니다.
    evalset_loader = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size, shuffle=False,
                                                 collate_fn=trainset.collate_fn, num_workers=args.num_workers,
                                                 pin_memory=True, drop_last=False)

    # test 데이터셋 및 로더
    testset = EOGTestDataset(root=data_root, split='test', padding_length=padding_length, args=args)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 collate_fn=testset.collate_fn, num_workers=args.num_workers,
                                                 pin_memory=True, drop_last=False)

    # 로깅을 위한 리스트 초기화
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    lrs = []

    # 모델생성
    num_heads = args.num_heads
    cnn_hdden_size = args.cnn_hdden_size
    lstm_hdden_size = args.lstm_hdden_size
    num_layers_lstm = args.num_layers_lstm
    dropout_lstm = args.dropout_lstm

    # Model selection based on argument
    if args.model_name == 'EOG1DCNN':
        model = EOG1DCNN(input_dim=args.input_dim, hidden_dim=cnn_hdden_size, num_classes=args.num_classes)
    elif args.model_name == 'EOGLSTMOneWay':
        model = EOGLSTMOneWay(input_dim=args.input_dim, hidden_dim=lstm_hdden_size, num_layers=num_layers_lstm, num_classes=args.num_classes, dropout=dropout_lstm)
    elif args.model_name == 'EOGLSTMNoAttention':
        model = EOGLSTMNoAttention(input_dim=args.input_dim, hidden_dim=lstm_hdden_size, num_layers=num_layers_lstm, num_classes=args.num_classes,
                                           dropout=dropout_lstm)
    elif args.model_name == 'EOGLSTMAttention':
        model = EOGLSTMAttention(input_dim=args.input_dim, hidden_dim=lstm_hdden_size, num_layers=num_layers_lstm, num_classes=args.num_classes,
                                num_heads=num_heads, dropout=dropout_lstm)

    best_val_loss = float('inf')
    val_loss = best_val_loss  # 이 값을 적절한 초기값으로 설정

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # optimizer 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40, verbose=True, min_lr=1e-6)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=0.5, gamma=2.0)
    # criterion = FocalTverskyLoss(alpha=0.99, beta=0.01, gamma=3, smooth=1e-6, reduction='mean')

    criterion_name = 'criterion: CrossEntropyLoss'
    # criterion_name = 'criterion: FocalTverskyLoss((alpha=0.8, beta=0.2, gamma=3, smooth=1e-6, reduction=mean)'
    # criterion_name = 'criterion: FocalLoss'

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_one_epoch(args, model,
                                                                                              optimizer,
                                                                                              criterion,
                                                                                              trainset_loader,
                                                                                              lr_scheduler, scaler,
                                                                                              args.accumulation_steps,
                                                                                              device, epoch)

        # running_loss / len(trainset_loader), accuracy, precision, recall, f1
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)  # 학습률 기록

        # Train 결과 저장
        with open(train_results_file, "a") as f:
            f.write(
                f"{epoch}\t{train_loss:.4f}\t{train_accuracy:.2f}\t{train_precision:.4f}\t{train_recall:.4f}\t{train_f1:.4f}\n")

        # --- Validation Phase ---
        last_valid_val_loss = None
        if epoch % args.initial_validation_interval == 0:
            model.eval()
            # evaluate 함수 호출: 제거 전과 후의 성능을 계산
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_cm, removed_cases = evaluate(
                args, model, evalset_loader, device, criterion, criterion_name, epoch,
                num_incorrect_to_remove=args.num_incorrect_to_remove
            )

            # # val_loss가 None이 아닌 경우에만 처리
            if val_loss is not None:
                last_valid_val_loss = val_loss  # 유효한 val_loss 업데이트
                val_losses.append(val_loss)  # 검증 손실 기록
                val_accuracies.append(val_accuracy)  # 검증 정확도 기록
                # print(val_accuracies)

                # Validation 결과 저장
                with open(val_results_file, "a") as f:
                    f.write(
                        f"{epoch}\t{val_loss:.4f}\t{val_accuracy:.2f}\t{val_precision:.4f}\t{val_recall:.4f}\t{val_f1:.4f}\n")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_weight_file_path)

            else:
                # val_loss가 None인 경우, 처리 방법을 결정합니다.
                # 예: 로그를 남기거나 특정 기본값을 설정할 수 있습니다.
                print(
                    "Validation loss is None, skipping this epoch's validation metrics comparison and model checkpointing.")

            torch.save(model.state_dict(), last_weight_file_path)

        # # 스케줄러 step 호출
        # lr_scheduler.step(val_loss)
        # 스케줄러 step 호출, 마지막 유효한 val_loss 사용
        if last_valid_val_loss is not None:
            lr_scheduler.step(last_valid_val_loss)
        else:
            print("No valid validation loss available for LR scheduler step.")

        # --- Test Phase ---
        if epoch % args.initial_inference_interval == 0:
            model.eval()  # 모델을 평가 모드로 설정

            outputs, labels, final_accuracy, final_precision, final_recall, final_f1, confusion_matrix, removed_cases = inference(
                args, model, testset_loader, device, criterion,
                num_incorrect_to_remove=args.num_incorrect_to_remove,
                epoch=epoch
            )

            # Test model 저장
            test_weight_file_path = os.path.join(ab_output_dir, f'{epoch}_test_model.pth')
            torch.save(model.state_dict(), test_weight_file_path)
            # Test 결과 저장
            with open(test_results_file, "a") as f:
                f.write(f"{epoch}\t{final_accuracy:.2f}\t{final_precision:.4f}\t{final_recall:.4f}\t{final_f1:.4f}\n")

            # 결과를 .txt 파일에 저장하기
            test_output_file_path_dir = os.path.join('/datasets', args.output_dir)
            test_output_file_path = os.path.join(test_output_file_path_dir, f"test_{epoch}_outputs.txt")
            if not os.path.exists(test_output_file_path_dir):
                os.makedirs(test_output_file_path_dir)

            try:
                with open(test_output_file_path, "w") as f:
                    f.write("Output\tLabel\n")
                    for output_value, label_value in zip(outputs, labels):
                        f.write(f"{output_value}\t{label_value}\n")

                    # 제거 후 성능을 저장
                    f.write(f"\nFinal Average Accuracy: {final_accuracy:.2f}%\n")
                    f.write(f"Final Precision: {final_precision:.4f}\n")
                    f.write(f"Final Recall: {final_recall:.4f}\n")
                    f.write(f"Final F1 Score: {final_f1:.4f}\n")
                    f.write(f"Final Confusion Matrix:\n{confusion_matrix}\n")
                    if args.num_incorrect_to_remove > 0:
                        f.write("\nRemoved Incorrect Predictions:\n")
                        for case in removed_cases:
                            predicted_label, true_label, path, _ = case
                            f.write(f"Path: {path}, True Label: {true_label}, Predicted Label: {predicted_label}\n")
            except Exception as e:
                print(f"Failed to save inference results: {e}")

        if epoch % 50 == 0 or epoch == args.epochs:
            plot_filename = os.path.join('/datasets', args.output_dir,
                                         f"epoch_{epoch}_training_validation_lr_curve.png")
            if not os.path.exists(os.path.join('/datasets', args.output_dir)):
                os.makedirs(os.path.join('/datasets', args.output_dir))

            save_plots(train_losses, val_losses, train_accuracies, val_accuracies, lrs, filename_prefix=plot_filename)

    final_plot_filename = os.path.join('/datasets', args.output_dir, "final_training_validation_lr_curve.png")
    save_plots(train_losses, val_losses, train_accuracies, val_accuracies, lrs, filename_prefix=final_plot_filename)

    model.train()
