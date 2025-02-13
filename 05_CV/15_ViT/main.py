import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn.functional as F

pl.seed_everything(42) 

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.backends.cudnn.deterministic = True  # GPU 연산에서 결정적 결과를 보장
torch.backends.cudnn.benchmark = False  # 최적화 시 CUDA 벤치마크를 사용하지 않음 (결정적 결과를 위해)


import pytorch_lightning as pl
import torch.optim as optim 
from model import VisionTransformer
from dataload import data_load

train_loader, val_loader, test_loader=data_load()

class ViT(pl.LightningModule):
    
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]   
    
    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


import os 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint 
import pytorch_lightning as pl
import torch
CHECKPOINT_PATH = "../saved_models/tutorial15" 
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train_model(**kwargs):
    # 트레이너를 설정합니다. 체크포인트 저장 위치와 사용할 장치를 지정합니다.
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"), 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=500,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # True이면 텐서보드에서 계산 그래프를 표시합니다.
    trainer.logger._default_hp_metric = None # 필요하지 않은 로깅 인자입니다.

    # 사전 학습된 모델이 있는지 확인하고, 있으면 불러와서 학습을 건너뜁니다.
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"사전 학습된 모델을 {pretrained_filename}에서 찾았습니다. 로딩 중...")
        model = ViT.load_from_checkpoint(pretrained_filename) # 저장된 하이퍼파라미터로 모델을 자동으로 불러옵니다.
    else:
        pl.seed_everything(42) # 재현 가능하게 하기 위해 시드 설정
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # 학습 후 최고의 체크포인트를 불러옵니다.

    # 검증 세트와 테스트 세트에서 모델을 평가합니다.
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result



def main():




    model, results = train_model(model_kwargs={
                                'embed_dim': 256,
                                'hidden_dim': 512,
                                'num_heads': 8,
                                'num_layers': 6,
                                'patch_size': 4,
                                'num_channels': 3,
                                'num_patches': 64,
                                'num_classes': 10,
                                'dropout': 0.2,
                                'train_loader':train_loader,
                                'val_loader':val_loader,
                                'test_loader':test_loader
                            },
                            lr=3e-4)
    print("ViT results", results)

    return 0


if __name__ == '__main__':
    main()