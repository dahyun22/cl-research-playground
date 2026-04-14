"""
파인튜닝 기준선 트레이너
==============================
가장 단순한 기준선 방법입니다.
현재 태스크 데이터만 사용해 일반적인 SGD/Adam 방식으로 학습합니다.
추가 메모리 사용이나 정규화 기법은 없습니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class FinetuneTrainer:
    """
    파인튜닝 기준선 트레이너입니다.

    기본 아이디어:
      - 각 새 태스크를 표준 cross-entropy loss로 학습
      - catastrophic forgetting을 막는 별도 장치는 없음
      - 다른 방법들과 비교하기 위한 기준선 역할
      - 이전 태스크 성능 저하가 크게 나타날 것으로 예상됨
    """
    
    def __init__(self, model, device="cpu", learning_rate=0.001, epochs=5):
        """
        Args:
            model: 신경망 모델 (예: MNIST_MLP, CIFAR10_CNN)
            device (str): 학습에 사용할 장치 ("cpu" 또는 "cuda")
            learning_rate (float): 옵티마이저 학습률
            epochs (int): 태스크별 학습 epoch 수
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train_task(self, train_loader, verbose=False):
        """
        단일 태스크에 대해 모델을 학습합니다.

        Args:
            train_loader: 현재 태스크 학습 데이터용 DataLoader
            verbose (bool): 학습 진행 상황 출력 여부

        Returns:
            float: 마지막 epoch 기준 평균 학습 loss
        """
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for x, y, _ in train_loader:
                x, y = x.to(self.device), y.to(self.device) 
                
                self.optimizer.zero_grad() # 배치마다 그래디언트 초기화
                
                logits = self.model(x) 
                loss = self.criterion(logits, y) # cross-entropy loss 계산
                
                loss.backward() # 그래디언트 계산 = "어떤 파라미터가 이 오차에 얼마나 기여했는가"
                self.optimizer.step()  # 파라미터 업데이트 
                
                total_loss += loss.item() 
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, test_loader):
        """
        테스트 데이터에서 모델 정확도를 평가합니다.

        Args:
            test_loader: 테스트 데이터용 DataLoader

        Returns:
            float: 정확도 (0~1)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits = self.model(x)
                preds = torch.argmax(logits, dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / max(total, 1)
        return accuracy
