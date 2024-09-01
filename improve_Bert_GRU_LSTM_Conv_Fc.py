import torch
import torch.nn as nn

class Bert_GRU_LSTM_Conv_fc(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(Bert_GRU_LSTM_Conv_fc, self).__init__()

        self.bert_model = bert_model

        self.lstm_1 = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bias=True)
        self.lstm_2 = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bias=True)
        self.lstm_3 = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bias=True)
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=1, batch_first=True, bias=True)

        self.conv_1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.conv_1_1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_2_2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_3_3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_4_4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(p=0.3)

        self.flatten = nn.Flatten()

        self.max_pooling = nn.MaxPool1d(2, 2, 0)
        
        self.gelu = nn.GELU()

        self.fc1 = nn.Linear(in_features=65536, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=128, bias=True)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input_ids, attention_mask, labels):
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        lstm1_out, _ = self.lstm_1(bert_output)
#         lstm1_out = torch.relu(lstm1_out)
#         lstm1_out = self.dropout(lstm1_out)

        lstm2_out, _ = self.lstm_2(bert_output)
#         lstm2_out = torch.relu(lstm2_out)
#         lstm2_out = self.dropout(lstm2_out)

        lstm3_out, _ = self.lstm_3(bert_output)
#         lstm3_out = self.dropout(lstm3_out)

        gru_out, _ = self.gru(bert_output)

        conv1_out = self.conv_1(lstm1_out.permute(0, 2, 1))
        conv1_out = self.dropout(conv1_out)

        conv2_out = self.conv_2(lstm2_out.permute(0, 2, 1))
        conv2_out = torch.relu(conv2_out)

        conv3_out = self.conv_3(lstm3_out.permute(0, 2, 1))
        conv3_out = self.dropout(conv3_out)

        conv4_out = self.conv_4(gru_out.permute(0, 2, 1))
        conv4_out = torch.relu(conv4_out)

        conv1_1_out = self.conv_1_1(conv1_out)
        conv1_1_out = self.dropout(conv1_1_out)

        conv2_2_out = self.conv_2_2(conv2_out)
        conv2_2_out = self.dropout(conv2_2_out)

        conv3_3_out = self.conv_3_3(conv3_out)  
        conv3_3_out = torch.relu(conv3_3_out) 

        conv4_4_out = self.conv_4_4(conv4_out)
        conv4_4_out = torch.relu(conv4_4_out)
        conv4_4_out = self.dropout(conv4_4_out)

        x_out = torch.cat((conv1_1_out, conv2_2_out, conv3_3_out, conv4_4_out), dim=1)

        x_out = self.max_pooling(x_out)

        x = self.flatten(x_out)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits