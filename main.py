import argparse
from train import train_models
from predict import optimize_and_predict

def main():
    parser = argparse.ArgumentParser(description="Dust concentration prediction and optimization.")
    
    # 添加参数
    parser.add_argument('--mode', required=True, choices=['train', 'predict'], help='Mode to run the script in: "train" to train models, "predict" to run predictions.')
    parser.add_argument('--dust_file', default='粉尘数据all.csv', help='Path to the dust concentration CSV file (used in "train" mode).')
    parser.add_argument('--b_file', default='粉尘数据-b.csv', help='Path to the B value CSV file (used in "train" mode).')
    parser.add_argument('--save_dir', default='./models', help='Directory to save trained models and scalers (used in "train" mode).')
    parser.add_argument('--fold', type=int, default=1, help='Fold number to use for model prediction (used in "predict" mode).')
    parser.add_argument('--max_epochs', type=int, default=5000, help='Maximum number of epochs for training (used in "train" mode).')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience (used in "train" mode).')
    args = parser.parse_args()

    # 执行模式
    if args.mode == 'train':
        print("\n--- 开始训练模型 ---\n")
        train_models(
            dust_file=args.dust_file,
            b_file=args.b_file,
            save_dir=args.save_dir,
            max_epochs=args.max_epochs,
            patience=args.patience
        )
        print("\n--- 模型训练完成 ---\n")
    elif args.mode == 'predict':
        print("\n--- 开始粉尘浓度预测与优化 ---\n")
        optimize_and_predict()  # 优化函数已经内置用户输入
        print("\n--- 预测与优化完成 ---\n")

if __name__ == "__main__":
    main()
