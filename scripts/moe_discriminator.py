import torch
import torch.nn as nn
import torch.nn.functional as F

class RowAttentionPool(nn.Module):
    """
    行级注意力池化：
    - 输入：x (B, C, H, W)
    - 输出：row_feats (B, C, H)，每一行一个 C 维表示
    通过一个 1x1 conv 产生打分，再对每一行的 W 维做 softmax 归一化作为注意力权重。
    """
    def __init__(self, in_channels: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1)  # (B,1,H,W)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        logits = self.score(self.drop(x))          # (B,1,H,W)
        attn = F.softmax(logits, dim=-1)           # 对 W 维做 softmax
        # 加权求和： (B,C,H,W) * (B,1,H,W) -> sum_W -> (B,C,H)
        row_feats = (x * attn).sum(dim=-1)
        return row_feats  # (B, C, H)


class Discriminator(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        
        # 行级特征池化
        self.row_attention_pool = RowAttentionPool(channels, dropout=dropout)
        
        # 若通道数不为1，可能需要一个卷积层来处理特征图
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # 输出一个概率
        )

    def forward(self, x):
        # 对每一行进行信息提取
        x = self.conv(x)  # 可以使用卷积提取每行的特征
        
        # 通过行注意力池化提取每一行的特征表示
        row_feats = self.row_attention_pool(x)  # (B, C, H)
        
        # 将行特征展平，并通过全连接层进行分类
        row_feats_flat = row_feats.mean(dim=-1)  # 将每行的特征平均（或根据任务选择其他方式）
        
        # 最终输出二分类的概率
        out = self.fc(row_feats_flat)  # (B, 1)
        return torch.sigmoid(out)  # 返回概率值（0 或 1）


# 示例：初始化判别器并执行前向传播
if __name__ == "__main__":
    B, C, H, W = 8, 64, 32, 32  # Batch size, Channels, Height, Width
    x = torch.randn(B, C, H, W)  # 模拟一个输入
    model = Discriminator(channels=C)
    output = model(x)
    print(output.shape)  # 输出一个形状为 (B, 1) 的概率
