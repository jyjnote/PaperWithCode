from data_download_load import down_and_load
from model import *  # 모델 정의를 가져옵니다.
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train import train_val

def main():
    # TensorBoard Writer 설정
    writer = SummaryWriter('runs/experiment1')

    # 데이터 로드
    train_ds, val_ds = down_and_load()

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=True)
    # 모델, 손실 함수 및 옵티마이저 설정

    model = InceptionResNetV2(10, 20, 10).to("cuda")
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=0.001)


    device="cuda"

    # x = torch.randn((3, 3, 299, 299)).to(device)
    # model = Stem().to(device)
    # output_Stem = model(x)

    # model = Inception_Resnet_A(output_Stem.size()[1]).to(device)
    # output_resA = model(output_Stem)

    # model = ReductionA(output_resA.size()[1], 256, 256, 384, 384).to(device)
    # output_rA = model(output_resA)

    # model = Inception_Resnet_B(output_rA.size()[1]).to(device)
    # output_resB = model(output_rA)

    # model = ReductionB(output_resB.size()[1]).to(device)
    # output_rB = model(output_resB)

    # model = Inception_Resnet_C(output_rB.size()[1]).to(device)
    # output_resC = model(output_rB)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

    params_train = {
    'num_epochs':50,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':val_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
    }

    model, loss_hist, metric_hist = train_val(model, params_train)


    # Train-Validation Progress
    num_epochs=params_train["num_epochs"]

    # 훈련 결과를 TensorBoard에 기록
    for epoch in range(params_train['num_epochs']):
        writer.add_scalar('Loss/train', loss_hist["train"][epoch], epoch)
        writer.add_scalar('Loss/val', loss_hist["val"][epoch], epoch)
        writer.add_scalar('Accuracy/train', metric_hist["train"][epoch], epoch)
        writer.add_scalar('Accuracy/val', metric_hist["val"][epoch], epoch)

        # 학습률 기록
        current_lr = opt.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

    writer.close()


if __name__ == '__main__':
    main()