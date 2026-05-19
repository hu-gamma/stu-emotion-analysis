"""
V8 多轮对话情感预测 CLI

用法:
    python scripts/eval/predict_multiturn.py --text "今天很开心。但是明天要考试了。"
    python scripts/eval/predict_multiturn.py --file input.txt
    python scripts/eval/predict_multiturn.py --text "..." --html
    python scripts/eval/predict_multiturn.py --interactive
"""
import sys
import argparse
sys.path.insert(0, '/mnt')

import torch
from models.model_v8_multiturn import RoBERTaBiLSTMV8MultiTurn

EMOTION_COLORS = {
    '悲伤': '\033[94m',   # blue
    '快乐': '\033[93m',   # yellow
    '愤怒': '\033[91m',   # red
    '焦虑': '\033[95m',   # magenta
    '厌恶': '\033[92m',   # green
    '惊讶': '\033[96m',   # cyan
    '好奇': '\033[36m',   # dark cyan
}
RESET = '\033[0m'


def load_model(model_path, device):
    model = RoBERTaBiLSTMV8MultiTurn(
        num_classes=7, dropout_rate=0.3,
        num_fusion_layers=4, fusion_mode='weighted_sum')
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def print_colored(results):
    for i, r in enumerate(results):
        color = EMOTION_COLORS.get(r['emotion'], '')
        bar = '█' * int(r['confidence'] * 20)
        print(f"  {color}[{r['emotion']}]{RESET} {bar} {r['confidence']:.0%}")
        print(f"    {r['sentence']}")
        if r['prev_context']:
            print(f"    \033[90m← 上文: {r['prev_context'][:60]}...\033[0m")
        print()


def main():
    parser = argparse.ArgumentParser(description='V8 多轮对话情感预测')
    parser.add_argument('--model', '-m', type=str,
                        default='/mnt/checkpoints/roberta_bilstm_v8_7class_L4_d03_lw/best_model.pt')
    parser.add_argument('--text', '-t', type=str, help='输入文本')
    parser.add_argument('--file', '-f', type=str, help='从文件读取文本')
    parser.add_argument('--html', action='store_true', help='输出 HTML 可视化')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互模式')
    parser.add_argument('--context-window', '-w', type=int, default=1, help='上下文窗口大小')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"加载模型: {args.model}")
    model = load_model(args.model, device)
    print(f"设备: {device}\n")

    if args.interactive:
        print("多轮对话情感预测 - 交互模式")
        print("输入文本（多句），回车预测，输入 'quit' 退出\n")
        while True:
            text = input("> ").strip()
            if text.lower() in ('quit', 'exit', 'q'):
                break
            if not text:
                continue
            results = model.predict_multiturn(text, device, args.context_window)
            print_colored(results)

    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        print(f"输入文本 ({len(text)} 字符):\n{text}\n")
        print("=" * 60)
        results = model.predict_multiturn(text, device, args.context_window)

        if args.html:
            html, _ = model.predict_multiturn_html(text, device, args.context_window)
            print(html)
        else:
            print_colored(results)
            print("\n情感序列:", ' → '.join([r['emotion'] for r in results]))

    elif args.text:
        print(f"输入: {args.text}\n")
        print("=" * 60)
        results = model.predict_multiturn(args.text, device, args.context_window)

        if args.html:
            html, _ = model.predict_multiturn_html(args.text, device, args.context_window)
            print(html)
        else:
            print_colored(results)
            print("情感序列:", ' → '.join([r['emotion'] for r in results]))

    else:
        # Demo mode
        demos = [
            "今天老师在全班面前表扬了我，真的好开心！不过下周就要期末考试了，有点紧张。希望能考好一点。",
            "又被室友吵醒了，真的很烦。为什么总是这样？我真的受够了。",
            "今天食堂的饭好难吃，而且排队排了好久，心情很差。",
        ]
        print("Demo 模式:\n")
        for text in demos:
            print(f"输入: {text}")
            results = model.predict_multiturn(text, device, args.context_window)
            print_colored(results)


if __name__ == '__main__':
    main()
