import numpy as np
import librosa
import sounddevice as sd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gradio as gr
import os
import matplotlib.pyplot as plt
import logging
import platform
import tempfile
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
model_path = Path(__file__).parent / "wake_word_model.pth"

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device_index = 0
if platform.system() == "Darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device(f"mps:{device_index}")
    logger.info(f"Using MPS device: {device}")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{device_index}")
    logger.info(f"Using CUDA device: {device}")
else:
    device = torch.device("cpu")
    logger.info("GPU acceleration not available, using CPU")
print(f"使用设备: {device}")

# 修正特征维度常量
FEATURE_DIM = 84  # 40 MFCC + 40 Delta + 4其他特征（修正）

# 定义更复杂的Siamese网络
class SiameseNetwork(nn.Module):
    def __init__(self, input_size=FEATURE_DIM, hidden_size=256, embedding_size=128):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.Tanh()  # 使用Tanh限制输出范围
        )
        
    def forward(self, x):
        output = self.embedding_net(x)
        return output
    
    def get_embedding(self, x):
        return self.forward(x)

# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# 自定义数据集
class WakeWordDataset(Dataset):
    def __init__(self, wake_word_files, background_files=None, sample_rate=16000, n_mfcc=40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.wake_word_features = []
        
        # 加载唤醒词样本
        for file in wake_word_files:
            if os.path.exists(file):
                try:
                    audio, _ = librosa.load(file, sr=sample_rate)
                    # 音频预处理：归一化
                    audio = self.preprocess_audio(audio)
                    feature = self.extract_features(audio)
                    # 检查特征是否有效且维度正确
                    if (not np.all(feature == 0) and 
                        not np.any(np.isnan(feature)) and 
                        len(feature) == FEATURE_DIM):
                        self.wake_word_features.append(feature)
                        logger.info(f"成功加载唤醒词样本: {file}, 特征维度: {len(feature)}")
                    else:
                        logger.warning(f"无效的唤醒词样本特征: {file}, 维度: {len(feature)}")
                except Exception as e:
                    logger.error(f"加载唤醒词样本失败 {file}: {e}")
        
        # 如果唤醒词样本不足，使用数据增强
        if len(self.wake_word_features) < 2:
            self.augment_wake_word_features()
        
        # 加载背景噪音样本（负样本）
        self.background_features = []
        if background_files:
            for file in background_files:
                if os.path.exists(file):
                    try:
                        audio, _ = librosa.load(file, sr=sample_rate)
                        audio = self.preprocess_audio(audio)
                        feature = self.extract_features(audio)
                        if (not np.all(feature == 0) and 
                            not np.any(np.isnan(feature)) and 
                            len(feature) == FEATURE_DIM):
                            self.background_features.append(feature)
                            logger.info(f"成功加载背景样本: {file}, 特征维度: {len(feature)}")
                        else:
                            logger.warning(f"无效的背景样本特征: {file}, 维度: {len(feature)}")
                    except Exception as e:
                        logger.error(f"加载背景样本失败 {file}: {e}")
        
        # 如果没有提供背景噪音或数量不足，生成一些真实感的噪音作为负样本
        if len(self.background_features) < 10:
            self.generate_background_features()
        
        # 确保所有特征维度一致
        self._validate_feature_dimensions()
        
        logger.info(f"数据集统计 - 唤醒词样本: {len(self.wake_word_features)}, 背景样本: {len(self.background_features)}")
    
    def preprocess_audio(self, audio):
        """音频预处理：归一化"""
        if len(audio) > 0:
            audio = audio / np.max(np.abs(audio) + 1e-8)  # 避免除零
        return audio
    
    def _validate_feature_dimensions(self):
        """验证所有特征维度是否一致"""
        for i, feat in enumerate(self.wake_word_features):
            if len(feat) != FEATURE_DIM:
                logger.warning(f"唤醒词特征 {i} 维度不正确: {len(feat)}，将进行裁剪/填充")
                self.wake_word_features[i] = self._fix_feature_dimension(feat)
        
        for i, feat in enumerate(self.background_features):
            if len(feat) != FEATURE_DIM:
                logger.warning(f"背景特征 {i} 维度不正确: {len(feat)}，将进行裁剪/填充")
                self.background_features[i] = self._fix_feature_dimension(feat)
    
    def _fix_feature_dimension(self, feature):
        """修复特征维度"""
        if len(feature) < FEATURE_DIM:
            # 填充
            return np.pad(feature, (0, FEATURE_DIM - len(feature)), mode='constant')
        else:
            # 裁剪
            return feature[:FEATURE_DIM]
    
    def augment_wake_word_features(self):
        """改进的数据增强：通过音频变换生成更多样本"""
        logger.info("正在进行数据增强...")
        original_features = self.wake_word_features.copy()
        
        for feature in original_features:
            # 添加高斯噪声
            noise = np.random.normal(0, 0.02, feature.shape)
            augmented_feature = feature + noise
            self.wake_word_features.append(augmented_feature)
            
            # 随机缩放
            scale = np.random.uniform(0.8, 1.2)
            scaled_feature = feature * scale
            self.wake_word_features.append(scaled_feature)
            
            # 时间偏移
            shift = np.random.randint(-8, 8)
            shifted_feature = np.roll(feature, shift)
            self.wake_word_features.append(shifted_feature)
            
            # 频率偏移（模拟音高变化）
            freq_shift = np.random.uniform(0.9, 1.1)
            freq_shifted_feature = feature * freq_shift
            self.wake_word_features.append(freq_shifted_feature)
        
        logger.info(f"数据增强后，唤醒词样本数量: {len(self.wake_word_features)}")
    
    def generate_background_features(self):
        """生成更真实的背景噪音特征"""
        logger.info("生成更真实的背景噪音特征...")
        num_background = max(30, len(self.wake_word_features) * 5)  # 增加负样本数量
        
        # 基于真实唤醒词特征生成负样本
        for wake_feature in self.wake_word_features:
            for _ in range(4):  # 每个正样本生成4个负样本
                modified_feature = wake_feature.copy()
                
                # 多种修改方式
                modification_type = np.random.choice(['shuffle', 'noise', 'scale', 'shift', 'invert'])
                
                if modification_type == 'shuffle':
                    np.random.shuffle(modified_feature)
                elif modification_type == 'noise':
                    noise = np.random.normal(0, 0.15, modified_feature.shape)
                    modified_feature += noise
                elif modification_type == 'scale':
                    scale = np.random.uniform(0.3, 2.5)
                    modified_feature *= scale
                elif modification_type == 'shift':
                    shift = np.random.randint(-15, 15)
                    modified_feature = np.roll(modified_feature, shift)
                elif modification_type == 'invert':
                    modified_feature = -modified_feature
                
                self.background_features.append(modified_feature)
        
        # 添加各种类型的随机噪音
        for _ in range(num_background):
            noise_type = np.random.choice(['realistic', 'white', 'pink', 'brown'])
            
            if noise_type == 'realistic':
                # 模拟真实环境噪音
                base = np.random.randn(FEATURE_DIM) * 0.08
                speech_pattern = np.sin(np.linspace(0, 3*np.pi, FEATURE_DIM)) * 0.15
                noise = base + speech_pattern + np.random.randn(FEATURE_DIM) * 0.05
            elif noise_type == 'white':
                noise = np.random.randn(FEATURE_DIM) * 0.12
            elif noise_type == 'pink':
                # 粉红噪音
                noise = np.random.randn(FEATURE_DIM) * np.linspace(1, 0.1, FEATURE_DIM)
            else:  # brown
                # 布朗噪音
                noise = np.cumsum(np.random.randn(FEATURE_DIM)) * 0.1
            
            self.background_features.append(noise)
        
        logger.info(f"生成的背景噪音特征数量: {len(self.background_features)}")
    
    def extract_features(self, audio_data):
        """提取更丰富的音频特征，确保维度一致"""
        if len(audio_data) < self.sample_rate:
            audio_data = np.pad(audio_data, (0, max(0, self.sample_rate - len(audio_data))), mode='constant')
        else:
            audio_data = audio_data[:self.sample_rate]
        
        # 提取多种特征
        features = []
        
        try:
            # 1. MFCC特征
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfccs.T, axis=0)
            features.extend(mfcc_mean)
            
            # 2. MFCC的一阶差分
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
            features.extend(mfcc_delta_mean)
            
            # 3. 频谱质心
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            features.append(np.mean(spectral_centroid))
            
            # 4. 过零率
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            features.append(np.mean(zero_crossing_rate))
            
            # 5. 频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
            features.append(np.mean(spectral_bandwidth))
            
            # 6. 频谱滚降点
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            features.append(np.mean(spectral_rolloff))
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 返回零特征
            return np.zeros(FEATURE_DIM)
        
        feature = np.array(features)
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保特征维度正确
        if len(feature) != FEATURE_DIM:
            logger.warning(f"特征维度不正确: {len(feature)}，期望: {FEATURE_DIM}")
            feature = self._fix_feature_dimension(feature)
        
        return feature
    
    def __len__(self):
        return max(len(self.wake_word_features), 5) * 20  # 增加样本数量
    
    def __getitem__(self, idx):
        # 改进的正负样本平衡生成
        if len(self.wake_word_features) >= 2 and np.random.random() < 0.5:
            # 正样本对：两个唤醒词样本
            idx1, idx2 = np.random.choice(len(self.wake_word_features), 2, replace=False)
            feature1 = self.wake_word_features[idx1]
            feature2 = self.wake_word_features[idx2]
            label = 0  # 相似
        else:
            # 负样本对
            if len(self.wake_word_features) > 0 and len(self.background_features) > 0:
                if np.random.random() < 0.7:
                    # 唤醒词 vs 背景噪音 (70%)
                    idx1 = np.random.randint(len(self.wake_word_features))
                    idx2 = np.random.randint(len(self.background_features))
                    feature1 = self.wake_word_features[idx1]
                    feature2 = self.background_features[idx2]
                else:
                    # 背景噪音 vs 背景噪音 (30%)
                    idx1, idx2 = np.random.choice(len(self.background_features), 2, replace=False)
                    feature1 = self.background_features[idx1]
                    feature2 = self.background_features[idx2]
                label = 1  # 不相似
            else:
                # 备用方案
                feature1 = np.random.randn(FEATURE_DIM) * 0.1
                feature2 = np.random.randn(FEATURE_DIM) * 0.1
                label = 1
        
        return (torch.FloatTensor(feature1), 
                torch.FloatTensor(feature2), 
                torch.FloatTensor([label]))

# 语音唤醒检测器
class AdvancedVoiceWakeWord:
    def __init__(self, sample_rate=16000, n_mfcc=40, threshold=0.7):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.threshold = threshold
        self.model = None
        self.wake_word_embeddings = []
        
    def train(self, wake_word_files, background_files=None, epochs=100, lr=0.0005):
        """训练模型"""
        # 检查是否有足够的唤醒词样本
        if not wake_word_files or len(wake_word_files) == 0:
            raise ValueError("没有提供唤醒词样本文件")
        
        # 创建数据集和数据加载器
        try:
            dataset = WakeWordDataset(wake_word_files, background_files, self.sample_rate, self.n_mfcc)
            
            if len(dataset.wake_word_features) < 2:
                raise ValueError("有效的唤醒词样本不足，无法创建正样本对")
                
            # 分割训练集和验证集 (80%训练, 20%验证)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
        except Exception as e:
            logger.error(f"创建数据集失败: {e}")
            raise
        
        # 初始化模型
        self.model = SiameseNetwork(input_size=FEATURE_DIM).to(device)
        criterion = ContrastiveLoss(margin=0.8)
        
        # 使用更稳定的优化器参数
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 使用余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=lr * 0.01
        )
        
        # 梯度裁剪
        max_grad_norm = 1.0
        
        # 训练模型
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_accuracy = 0
        patience_counter = 0
        patience = 25  # 增加早停耐心值
        
        self.model.train()
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0
            train_batch_count = 0
            train_correct = 0
            train_total = 0
            
            for feature1, feature2, label in train_loader:
                if feature1.shape[0] != feature2.shape[0]:
                    continue
                    
                feature1, feature2, label = feature1.to(device), feature2.to(device), label.to(device)
                
                optimizer.zero_grad()
                output1 = self.model(feature1)
                output2 = self.model(feature2)
                
                # 计算准确率
                with torch.no_grad():
                    distances = nn.functional.pairwise_distance(output1, output2)
                    predictions = (distances > 0.5).float()
                    train_correct += (predictions == label.squeeze()).sum().item()
                    train_total += label.size(0)
                
                loss = criterion(output1, output2, label)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                train_batch_count += 1
            
            # 验证阶段
            self.model.eval()
            epoch_val_loss = 0
            val_batch_count = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for feature1, feature2, label in val_loader:
                    if feature1.shape[0] != feature2.shape[0]:
                        continue
                        
                    feature1, feature2, label = feature1.to(device), feature2.to(device), label.to(device)
                    
                    output1 = self.model(feature1)
                    output2 = self.model(feature2)
                    
                    loss = criterion(output1, output2, label)
                    epoch_val_loss += loss.item()
                    val_batch_count += 1
                    
                    distances = nn.functional.pairwise_distance(output1, output2)
                    predictions = (distances > 0.5).float()
                    val_correct += (predictions == label.squeeze()).sum().item()
                    val_total += label.size(0)
            
            if train_batch_count > 0 and val_batch_count > 0:
                avg_train_loss = epoch_train_loss / train_batch_count
                avg_val_loss = epoch_val_loss / val_batch_count
                train_accuracy = train_correct / train_total if train_total > 0 else 0
                val_accuracy = val_correct / val_total if val_total > 0 else 0
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                
                # 更新学习率
                scheduler.step()
                
                # 早停机制
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    # 保存最佳模型
                    best_model_state = self.model.state_dict().copy()
                    logger.info(f"新的最佳验证准确率: {best_val_accuracy:.4f}")
                else:
                    patience_counter += 1
                
                if epoch % 5 == 0 or epoch == epochs - 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                               f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, LR: {current_lr:.6f}')
                
                # 早停检查
                if patience_counter >= patience:
                    logger.info(f"早停触发，最佳验证准确率: {best_val_accuracy:.4f}")
                    # 恢复最佳模型
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                train_losses.append(0)
                val_losses.append(0)
                train_accuracies.append(0)
                val_accuracies.append(0)
        
        # 计算唤醒词的参考嵌入
        self.model.eval()
        self.wake_word_embeddings = []
        for feature in dataset.wake_word_features:
            with torch.no_grad():
                embedding = self.model(torch.FloatTensor(feature).unsqueeze(0).to(device))
                self.wake_word_embeddings.append(embedding.cpu().numpy())
        
        # 绘制训练曲线
        self.plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # 分析训练曲线
        curve_analysis = self.analyze_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        logger.info(curve_analysis)
        
        return train_losses, curve_analysis
    
    def analyze_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """分析训练曲线"""
        if len(train_losses) < 2:
            return "训练轮数不足，无法分析曲线"
        
        analysis = "训练曲线分析:\n"
        analysis += "=" * 50 + "\n"
        
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1] if val_losses else 0
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        final_val_acc = val_accuracies[-1] if val_accuracies else 0
        
        analysis += f"最终训练损失: {final_train_loss:.4f}\n"
        analysis += f"最终验证损失: {final_val_loss:.4f}\n"
        analysis += f"最终训练准确率: {final_train_acc:.4f}\n"
        analysis += f"最终验证准确率: {final_val_acc:.4f}\n"
        
        # 过拟合检查
        if val_losses and len(val_losses) > 10:
            if val_losses[-1] > val_losses[-10] and train_losses[-1] < train_losses[-10]:
                analysis += "⚠️  检测到可能过拟合: 训练损失下降但验证损失上升\n"
        
        analysis += "\n诊断建议:\n"
        
        if final_train_loss > 0.5:
            analysis += "❌ 训练损失较高，建议增加训练轮数或调整学习率\n"
        elif final_train_loss > 0.2:
            analysis += "⚠️  训练损失适中，可以继续优化\n"
        else:
            analysis += "✅ 训练损失很低，训练效果良好\n"
        
        if final_val_acc < 0.7:
            analysis += "❌ 验证准确率较低，建议增加样本多样性或调整模型\n"
        elif final_val_acc < 0.85:
            analysis += "⚠️  验证准确率适中，还有提升空间\n"
        else:
            analysis += "✅ 验证准确率很高，训练成功\n"
        
        # 训练验证差距分析
        if train_accuracies and val_accuracies:
            gap = train_accuracies[-1] - val_accuracies[-1]
            if gap > 0.15:
                analysis += "⚠️  训练验证差距较大，可能存在过拟合\n"
        
        return analysis
    
    def plot_training_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """绘制训练指标"""
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if train_accuracies:
            plt.plot(train_accuracies, label='Training Accuracy')
        if val_accuracies:
            plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def is_wake_word(self, audio_data):
        """检测是否为唤醒词"""
        if self.model is None or not self.wake_word_embeddings:
            print("请先训练模型")
            return False, 0.0
        
        # 音频预处理
        audio_data = self.preprocess_audio(audio_data)
        
        # 提取特征
        feature = self.extract_features(audio_data)
        
        # 调试信息
        feature_mean = np.mean(feature)
        feature_std = np.std(feature)
        print(f"特征提取结果: 均值={feature_mean:.4f}, 标准差={feature_std:.4f}")
        
        # 获取嵌入向量
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature).unsqueeze(0).to(device)
            embedding = self.model(input_tensor).cpu().numpy()
        
        embedding_mean = np.mean(embedding)
        embedding_std = np.std(embedding)
        print(f"嵌入向量: 均值={embedding_mean:.4f}, 标准差={embedding_std:.4f}")
        
        # 计算欧氏距离（与训练时一致）
        distances = []
        for ref_embedding in self.wake_word_embeddings:
            distance = np.linalg.norm(embedding - ref_embedding)
            distances.append(distance)
        
        min_distance = min(distances) if distances else float('inf')
        avg_distance = np.mean(distances) if distances else float('inf')
        
        # 将距离转换为相似度分数
        similarity = 1.0 / (1.0 + min_distance)
        avg_similarity = 1.0 / (1.0 + avg_distance) if avg_distance != float('inf') else 0
        
        print(f"最小距离: {min_distance:.4f}, 平均距离: {avg_distance:.4f}")
        print(f"最大相似度: {similarity:.4f}, 平均相似度: {avg_similarity:.4f}")
        
        # 更严格的检测条件
        is_wake = (
            similarity > self.threshold and
            avg_similarity > self.threshold * 0.5 and
            min_distance < 0.8  # 距离阈值
        )
        
        return is_wake, similarity
    
    def preprocess_audio(self, audio_data):
        """音频预处理：归一化"""
        if len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            if max_val > 1e-8:  # 避免除零
                audio_data = audio_data / max_val
        return audio_data
    
    def extract_features(self, audio_data):
        """提取特征"""
        if len(audio_data) < self.sample_rate:
            audio_data = np.pad(audio_data, (0, max(0, self.sample_rate - len(audio_data))), mode='constant')
        else:
            audio_data = audio_data[:self.sample_rate]
        
        features = []
        
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfccs.T, axis=0)
            features.extend(mfcc_mean)
            
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
            features.extend(mfcc_delta_mean)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            features.append(np.mean(spectral_centroid))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            features.append(np.mean(zero_crossing_rate))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
            features.append(np.mean(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            features.append(np.mean(spectral_rolloff))
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return np.zeros(FEATURE_DIM)
        
        feature = np.array(features)
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保维度正确
        if len(feature) != FEATURE_DIM:
            feature = np.pad(feature, (0, max(0, FEATURE_DIM - len(feature))), mode='constant')[:FEATURE_DIM]
        
        return feature

# 处理上传的文件
def process_uploaded_files(files, target_dir):
    """处理上传的文件"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    saved_paths = []
    for file in files:
        if hasattr(file, 'name'):
            src_path = file.name
        else:
            src_path = file
            
        filename = os.path.basename(src_path)
        dest_path = os.path.join(target_dir, filename)
        shutil.copy2(src_path, dest_path)
        saved_paths.append(dest_path)
    
    return saved_paths

# Gradio UI函数
def train_model_ui(wake_word_files, background_files, epochs, lr, threshold):
    """Gradio训练界面"""
    try:
        if not wake_word_files:
            return "请先上传唤醒词样本", None, ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            wake_word_dir = os.path.join(temp_dir, "wake_word")
            background_dir = os.path.join(temp_dir, "background")
            
            wake_word_paths = process_uploaded_files(wake_word_files, wake_word_dir)
            background_paths = process_uploaded_files(background_files, background_dir) if background_files else None
            
            detector = AdvancedVoiceWakeWord(threshold=threshold)
            losses, curve_analysis = detector.train(wake_word_paths, background_paths, epochs=int(epochs), lr=lr)
            
            if detector.model is not None:
                torch.save({
                    'model_state_dict': detector.model.state_dict(),
                    'wake_word_embeddings': detector.wake_word_embeddings,
                    'threshold': threshold,
                    'input_size': FEATURE_DIM
                }, str(model_path))
                logger.info("模型已保存")
            
            return "训练完成！", 'training_metrics.png', curve_analysis
            
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        return f"训练失败: {str(e)}", None, ""

def detect_wake_word_ui(audio_file, threshold):
    """Gradio检测界面"""
    try:
        if audio_file is None:
            return "请先上传音频文件", 0.0
        
        if hasattr(audio_file, 'name'):
            audio_path = audio_file.name
        else:
            audio_path = audio_file
        
        detector = AdvancedVoiceWakeWord(threshold=threshold)
        
        # 改进的模型加载
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            except:
                # 兼容旧版本
                checkpoint = torch.load(model_path, map_location=device)
            
            input_size = checkpoint.get('input_size', FEATURE_DIM)
            detector.model = SiameseNetwork(input_size=input_size).to(device)
            detector.model.load_state_dict(checkpoint['model_state_dict'])
            detector.wake_word_embeddings = checkpoint['wake_word_embeddings']
            detector.model.eval()
        else:
            return "请先训练模型", 0.0
        
        audio, sr = librosa.load(audio_path, sr=16000)
        is_wake, confidence = detector.is_wake_word(audio)
        
        result_text = "✅ 检测到唤醒词！" if is_wake else "❌ 未检测到唤醒词"
        confidence_text = f"置信度: {confidence:.4f}"
        
        return f"{result_text}\n{confidence_text}", confidence
        
    except Exception as e:
        logger.error(f"检测过程中出错: {e}")
        return f"检测失败: {str(e)}", 0.0

def test_model_ui(wake_word_files, test_files, threshold):
    """测试模型性能"""
    try:
        if not wake_word_files or not test_files:
            return "请先上传唤醒词样本和测试样本"
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            wake_word_dir = os.path.join(temp_dir, "wake_word")
            test_dir = os.path.join(temp_dir, "test")
            
            # 处理上传的文件
            wake_word_paths = process_uploaded_files(wake_word_files, wake_word_dir)
            test_paths = process_uploaded_files(test_files, test_dir)
            
            # 加载模型
            detector = AdvancedVoiceWakeWord(threshold=threshold)
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                except:
                    checkpoint = torch.load(model_path, map_location=device)
                
                input_size = checkpoint.get('input_size', FEATURE_DIM)
                detector.model = SiameseNetwork(input_size=input_size).to(device)
                detector.model.load_state_dict(checkpoint['model_state_dict'])
                detector.wake_word_embeddings = checkpoint['wake_word_embeddings']
                detector.model.eval()
            else:
                return "请先训练模型"
            
            # 测试性能
            results = []
            confidences = []
            for test_file in test_paths:
                audio, sr = librosa.load(test_file, sr=16000)
                is_wake, confidence = detector.is_wake_word(audio)
                results.append(is_wake)
                confidences.append(confidence)
            
            # 生成测试报告
            total = len(results)
            correct = sum(results)
            accuracy = correct / total if total > 0 else 0
            avg_confidence = np.mean(confidences) if confidences else 0
            
            report = f"📊 测试结果:\n"
            report += f"总样本数: {total}\n"
            report += f"正确识别: {correct}\n"
            report += f"准确率: {accuracy:.4f}\n"
            report += f"平均置信度: {avg_confidence:.4f}\n\n"
            report += f"详细结果:\n"
            
            for i, (test_file, is_wake, confidence) in enumerate(zip(test_paths, results, confidences)):
                status = "✓" if is_wake else "✗"
                report += f"{i+1}. {os.path.basename(test_file)}: {status} (置信度: {confidence:.4f})\n"
            
            return report
            
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        return f"测试失败: {str(e)}"

# 创建Gradio界面
def create_gradio_ui():
    with gr.Blocks(title="改进版小样本语音唤醒训练系统") as demo:
        gr.Markdown("# 🎯 改进版小样本语音唤醒训练系统")
        
        with gr.Tab("训练模型"):
            gr.Markdown("## 训练语音唤醒模型")
            with gr.Row():
                with gr.Column():
                    wake_word_files = gr.File(file_count="multiple", label="上传唤醒词样本(WAV格式)", file_types=[".wav"])
                    background_files = gr.File(file_count="multiple", label="上传背景噪音样本(WAV格式，可选)", file_types=[".wav"])
                    epochs = gr.Slider(50, 300, value=120, step=10, label="训练轮数")
                    lr = gr.Slider(0.0001, 0.01, value=0.0005, step=0.0001, label="学习率")
                    threshold = gr.Slider(0.5, 0.95, value=0.75, step=0.05, label="检测阈值")
                    train_btn = gr.Button("🚀 开始训练", variant="primary")
                
                with gr.Column():
                    output_text = gr.Textbox(label="训练状态", lines=3)
                    output_plot = gr.Image(label="训练指标图表")
                    analysis_output = gr.Textbox(label="曲线分析", lines=10)
            
            train_btn.click(
                fn=train_model_ui,
                inputs=[wake_word_files, background_files, epochs, lr, threshold],
                outputs=[output_text, output_plot, analysis_output]
            )
        
        with gr.Tab("检测唤醒词"):
            gr.Markdown("## 检测音频中的唤醒词")
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(label="上传音频文件(WAV格式)", type="filepath")
                    detect_threshold = gr.Slider(0.5, 0.95, value=0.75, step=0.05, label="检测阈值")
                    detect_btn = gr.Button("🔍 开始检测", variant="primary")
                
                with gr.Column():
                    detect_output = gr.Textbox(label="检测结果", lines=3)
                    confidence_gauge = gr.Number(label="置信度", interactive=False)
            
            detect_btn.click(
                fn=detect_wake_word_ui,
                inputs=[audio_input, detect_threshold],
                outputs=[detect_output, confidence_gauge]
            )
        
        with gr.Tab("测试模型"):
            gr.Markdown("## 测试模型性能")
            with gr.Row():
                with gr.Column():
                    test_wake_word_files = gr.File(file_count="multiple", label="上传唤醒词样本", file_types=[".wav"])
                    test_files = gr.File(file_count="multiple", label="上传测试样本", file_types=[".wav"])
                    test_threshold = gr.Slider(0.5, 0.95, value=0.75, step=0.05, label="检测阈值")
                    test_btn = gr.Button("🧪 开始测试", variant="primary")
                
                with gr.Column():
                    test_output = gr.Textbox(label="测试结果", lines=15)
            
            test_btn.click(
                fn=test_model_ui,
                inputs=[test_wake_word_files, test_files, test_threshold],
                outputs=[test_output]
            )
        
        with gr.Tab("使用说明"):
            gr.Markdown("""
            ## 📖 使用说明
            
            ### 1. 训练模型
            - 上传至少2-5个清晰的唤醒词样本
            - 可选上传背景噪音样本提高模型鲁棒性
            - 建议参数: 训练轮数120-150, 学习率0.0005, 阈值0.75
            
            ### 2. 检测唤醒词
            - 先完成模型训练
            - 上传待检测的音频文件
            - 调整阈值控制检测灵敏度
            
            ### 3. 测试模型
            - 使用独立的测试集评估模型性能
            - 查看准确率和置信度指标
            
            ### 💡 最佳实践
            - 唤醒词样本时长1-2秒，发音清晰一致
            - 背景样本包含各种噪音和语音片段
            - 训练完成后使用测试集验证效果
            
            ### 🎯 改进特性
            - 修正了特征维度问题
            - 改进了数据增强和负样本生成
            - 添加了验证集和早停机制
            - 统一了距离度量标准
            - 增强了调试信息
            """)
    
    return demo

# 主函数
def main():
    # 创建Gradio界面并启动
    demo = create_gradio_ui()
    
    # 配置队列参数
    demo.queue(
        status_update_rate="auto",
        api_open=False,
        max_size=10
    )
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()