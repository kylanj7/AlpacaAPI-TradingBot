import torch
import torch.nn as nn

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(xLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Enhanced LSTM with layer normalization
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.lstm_layer_norm = nn.LayerNorm(hidden_size)
        
        #  attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),  # Lighter dropout for attention
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Multi-scale convolutions with proper normalization
        self.conv1 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=7, padding=3)
        
        # Batch normalization for conv layers
        self.conv_bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.conv_bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.conv_bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Enhanced gating mechanism
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_size + (hidden_size // 2) * 3, hidden_size),
            nn.LayerNorm(hidden_size),  # Replace with LayerNorm for stability
            nn.Tanh(),  # More stable than sigmoid
            nn.Dropout(dropout * 0.5)
        )
        
        # Residual connection preparation
        self.residual_projection = nn.Linear(hidden_size, hidden_size)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc1_norm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2_norm = nn.LayerNorm(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU often works better than ReLU

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM with layer normalization
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.lstm_layer_norm(lstm_out)

        #  attention with proper normalization
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_out = torch.sum(lstm_out * attention_weights, dim=1)

        # Enhanced multi-scale convolutions
        lstm_out_transposed = lstm_out.transpose(1, 2)
        
        conv1_out = self.activation(self.conv_bn1(self.conv1(lstm_out_transposed)))
        conv2_out = self.activation(self.conv_bn2(self.conv2(lstm_out_transposed)))
        conv3_out = self.activation(self.conv_bn3(self.conv3(lstm_out_transposed)))

        # Global max pooling for each conv output
        conv1_pooled = torch.max(conv1_out, dim=2)[0]
        conv2_pooled = torch.max(conv2_out, dim=2)[0]
        conv3_pooled = torch.max(conv3_out, dim=2)[0]
        
        multi_scale = torch.cat([conv1_pooled, conv2_pooled, conv3_pooled], dim=1)

        # Enhanced feature gating
        combined_features = torch.cat([attended_out, multi_scale], dim=1)
        gate_output = self.feature_gate(combined_features)
        
        # Residual connection
        residual = self.residual_projection(attended_out)
        gated_output = gate_output + residual  # Residual connection for stability

        # Enhanced output processing
        out = torch.cat([gated_output, attended_out], dim=1)
        
        # First FC layer with residual connection
        fc1_out = self.fc1(out)
        fc1_out = self.fc1_norm(fc1_out)
        fc1_out = self.activation(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # Add residual connection if dimensions match
        if fc1_out.size(1) == gated_output.size(1):
            fc1_out = fc1_out + gated_output
        
        # Second FC layer
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.fc2_norm(fc2_out)
        fc2_out = self.activation(fc2_out)
        fc2_out = self.dropout(fc2_out)
        
        # Final output
        output = self.fc3(fc2_out)
        
        return output