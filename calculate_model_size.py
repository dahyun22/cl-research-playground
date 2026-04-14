from models import MNIST_MLP

model = MNIST_MLP()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
size_mb_fp32 = total * 4 / (1024 ** 2)

print("params:", total)
print("trainable:", trainable)
print("fp32 MB:", round(size_mb_fp32, 3))

# GEM, DER++는 별도 메모리 버퍼가 있어서 모델 외 메모리 사용량이 늘어납니다
# Adam은 optimizer state 때문에 파라미터 메모리의 추가 배수가 붙습니다
# Co2L은 projection head가 있어서 기본 모델보다 약간 더 큽니다