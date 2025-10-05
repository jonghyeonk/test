import torch
import torch.nn as nn


class EOG1DCNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_classes=4):
        super(EOG1DCNN, self).__init__()
        # 1D CNN layers for VEOG and HEOG
        self.cnn_veog = nn.Conv1d(in_channels=input_dim * 2, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.cnn_heog = nn.Conv1d(in_channels=input_dim * 2, out_channels=hidden_dim, kernel_size=3, padding=1)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # VEOG + HEOG를 concat한 hidden_dim * 2

        # 사용하지 않는 매개변수 초기화 (인터페이스 통일을 위해 포함)
        print("EOG1DCNN model created")

    def forward(self, x):
        veog = x[:, :, 1].unsqueeze(1)  # (batch_size, 1, seq_len)
        heog = x[:, :, 0].unsqueeze(1)  # (batch_size, 1, seq_len)
        heog_diff = x[:, :, 2].unsqueeze(1)   # (B, 1, L)
        veog_diff = x[:, :, 3].unsqueeze(1)   # (B, 1, L)

        heog = torch.cat([heog, heog_diff], dim=1)  # (B, 2, L)
        veog = torch.cat([veog, veog_diff], dim=1)  # (B, 2, L)

        # Apply 1D CNNs
        veog_out = self.cnn_veog(veog).permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        heog_out = self.cnn_heog(heog).permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)

        # Global Average Pooling
        context_vector_veog = torch.mean(veog_out, dim=1)  # (batch_size, hidden_dim)
        context_vector_heog = torch.mean(heog_out, dim=1)  # (batch_size, hidden_dim)

        # Concatenate VEOG and HEOG context vectors
        combined_context = torch.cat([context_vector_veog, context_vector_heog], dim=-1)  # (batch_size, hidden_dim * 2)

        # Fully Connected Layer for class predictions
        output = self.fc(combined_context)  # (batch_size, num_classes)

        # Attention은 사용하지 않으므로 None 반환
        return output, (None, None)

class EOGLSTMOneWay(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, num_classes=4, dropout=0.1):
        super(EOGLSTMOneWay, self).__init__()
        # LSTM 초기화 (Bidirectional=False)
        self.lstm_veog = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=False, dropout=dropout)
        self.lstm_heog = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=False, dropout=dropout)

        # 최종 Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # VEOG + HEOG를 concat한 hidden_dim * 2
        print("EOGLSTMOneWay model created")
    def forward(self, x):
        veog = x[:, :, 1].unsqueeze(-1)  # Vertical EOG (batch_size, seq_len, 1)
        heog = x[:, :, 0].unsqueeze(-1)  # Horizontal EOG (batch_size, seq_len, 1)
        heog_diff = x[:, :, 2].unsqueeze(-1)  # (batch, seq_len, 1)
        veog_diff = x[:, :, 3].unsqueeze(-1)  # (batch, seq_len, 1)

        # H EOG 그룹 신호 결합
        heog = torch.cat([heog, heog_diff], dim=-1)  # (batch, seq_len, 2)
        # V EOG 그룹 신호 결합
        veog = torch.cat([veog, veog_diff], dim=-1)  # (batch, seq_len, 2)

        # LSTM의 출력
        lstm_output_veog, _ = self.lstm_veog(veog)  # (batch_size, seq_len, hidden_dim)
        lstm_output_heog, _ = self.lstm_heog(heog)  # (batch_size, seq_len, hidden_dim)

        # 시퀀스 길이에 대해 평균 (Global Average Pooling 역할)
        context_vector_veog = torch.mean(lstm_output_veog, dim=1)  # (batch_size, hidden_dim)
        context_vector_heog = torch.mean(lstm_output_heog, dim=1)  # (batch_size, hidden_dim)

        # VEOG와 HEOG를 결합
        combined_context = torch.cat([context_vector_veog, context_vector_heog], dim=-1)  # (batch_size, hidden_dim * 2)

        # 최종 클래스 예측
        output = self.fc(combined_context)  # (batch_size, num_classes)

        # Attention이 없으므로 두 번째 반환값은 None
        return output, (None, None)

class EOGLSTMNoAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, num_classes=4, dropout=0.1):
        super(EOGLSTMNoAttention, self).__init__()
        self.lstm_veog = nn.LSTM(input_size=input_dim*2, hidden_size=hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_heog = nn.LSTM(input_size=input_dim*2, hidden_size=hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self.dropout = nn.Dropout(dropout)  # Fully Connected Layer에 Dropout 추가

    def forward(self, x):
        veog = x[:, :, 1].unsqueeze(-1)
        heog = x[:, :, 0].unsqueeze(-1)
        heog_diff = x[:, :, 2].unsqueeze(-1)  # (batch, seq_len, 1)
        veog_diff = x[:, :, 3].unsqueeze(-1)  # (batch, seq_len, 1)

        # H EOG 그룹 신호 결합
        heog = torch.cat([heog, heog_diff], dim=-1)  # (batch, seq_len, 2)
        # V EOG 그룹 신호 결합
        veog = torch.cat([veog, veog_diff], dim=-1)  # (batch, seq_len, 2)

        lstm_output_veog, _ = self.lstm_veog(veog)
        lstm_output_heog, _ = self.lstm_heog(heog)

        avg_veog = torch.mean(lstm_output_veog, dim=1)
        avg_heog = torch.mean(lstm_output_heog, dim=1)

        combined_context = torch.cat([avg_veog, avg_heog], dim=-1)

        # Dropout 적용
        combined_context = self.dropout(combined_context)

        output = self.fc(combined_context)
        return output, (None, None)

class EOGLSTMAttention(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=4, num_heads=4, dropout=0.1):
        super(EOGLSTMAttention, self).__init__()
        self.lstm_heog = nn.LSTM(input_size=input_dim * 2,  # H_raw와 H_win_diff, H_first_diff 입력
                                 hidden_size=hidden_dim,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bidirectional=True)
        # V EOG 그룹 처리를 위한 LSTM (Raw + Diff)
        self.lstm_veog = nn.LSTM(input_size=input_dim * 2,  # V_raw와 V_diff, V_first_diff 입력
                                 hidden_size=hidden_dim,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bidirectional=True)

        embed_dim = hidden_dim * 2  # BiLSTM 출력 크기
        self.multihead_attn_veog = nn.MultiheadAttention(embed_dim=embed_dim,
                                                         num_heads=num_heads,
                                                         dropout=dropout,
                                                         batch_first=True)
        self.multihead_attn_heog = nn.MultiheadAttention(embed_dim=embed_dim,
                                                         num_heads=num_heads,
                                                         dropout=dropout,
                                                         batch_first=True)

        self.norm_veog = nn.LayerNorm(embed_dim)
        self.norm_heog = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(hidden_dim * 4, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, 4) -> [heog_raw, veog_raw, heog_diff, veog_diff]
        heog_raw = x[:, :, 0].unsqueeze(-1)  # (batch, seq_len, 1)
        veog_raw = x[:, :, 1].unsqueeze(-1)  # (batch, seq_len, 1)
        heog_window_diff = x[:, :, 2].unsqueeze(-1)  # (batch, seq_len, 1)
        veog_window_diff = x[:, :, 3].unsqueeze(-1)  # (batch, seq_len, 1)

        # H EOG 그룹 신호 결합
        heog_combined = torch.cat([heog_raw, heog_window_diff], dim=-1)  # (batch, seq_len, 2)
        # V EOG 그룹 신호 결합
        veog_combined = torch.cat([veog_raw, veog_window_diff], dim=-1)  # (batch, seq_len, 2)

        # LSTM에서 메모리 최적화를 위해 flatten_parameters 호출
        # 두 개의 LSTM만 관리하면 됩니다.
        self.lstm_heog.flatten_parameters()
        self.lstm_veog.flatten_parameters()

        lstm_output_veog, _ = self.lstm_veog(veog_combined)  # (B, L, 2*hidden)
        lstm_output_heog, _ = self.lstm_heog(heog_combined)

        # Multi-Head Attention (자기 자신을 Query/Key/Value로 사용)
        attn_output_veog, attn_weights_veog = self.multihead_attn_veog(
            lstm_output_veog, lstm_output_veog, lstm_output_veog
        )
        attn_output_heog, attn_weights_heog = self.multihead_attn_heog(
            lstm_output_heog, lstm_output_heog, lstm_output_heog
        )

        attn_weights_veog = attn_weights_veog.mean(dim=1)
        attn_weights_heog = attn_weights_heog.mean(dim=1)

        # Residual + LayerNorm
        attn_output_veog = self.norm_veog(lstm_output_veog + self.dropout(attn_output_veog))
        attn_output_heog = self.norm_heog(lstm_output_heog + self.dropout(attn_output_heog))

        # Context vector (sequence average pooling)
        context_vector_veog = torch.mean(attn_output_veog, dim=1)  # (B, 2*hidden)
        context_vector_heog = torch.mean(attn_output_heog, dim=1)  # (B, 2*hidden)

        # Concatenate
        combined_context = torch.cat([context_vector_veog, context_vector_heog], dim=-1)  # (B, 4*hidden)

        output = self.fc(combined_context)  # (B, num_classes)

        # Attention weights 반환 (H, V 각각)
        return output, (attn_weights_veog, attn_weights_heog)