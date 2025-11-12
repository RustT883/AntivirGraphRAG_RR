import re
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import argparse

def parse_log_file(log_file_path):
    """Read the entire file first, then apply regex for multi-line matching."""
    eval_pattern = re.compile(
        r'Step=(\d+).*?Micro P: ([\d.]+)%.*?Micro R: ([\d.]+)%.*?Micro F1: ([\d.]+)%'
        r'.*?Macro P: ([\d.]+)%.*?Macro R: ([\d.]+)%.*?Macro F1: ([\d.]+)%',
        re.DOTALL  # Allow .* to match newlines
    )
    
    data = {'train': [], 'eval': []}
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()  # Read entire file as a single string
    
    # Find all evaluation matches (multi-line)
    for eval_match in eval_pattern.finditer(log_content):
        step = int(eval_match.group(1))
        data['eval'].append({
            'step': step,
            'micro_precision': float(eval_match.group(2)),
            'micro_recall': float(eval_match.group(3)),
            'micro_f1': float(eval_match.group(4)),
            'macro_precision': float(eval_match.group(5)),
            'macro_recall': float(eval_match.group(6)),
            'macro_f1': float(eval_match.group(7))
        })
    
    # Find training loss (line-by-line is fine here)
    with open(log_file_path, 'r') as f:
        for line in f:
            train_match = re.search(r'Step (\d+) \| loss: ([\d.]+)', line)
            if train_match:
                data['train'].append({
                    'step': int(train_match.group(1)),
                    'loss': float(train_match.group(2))
                })
    
    return data

def write_to_tensorboard(data, log_dir):
    """Write all metrics to TensorBoard"""
    writer = SummaryWriter(log_dir=log_dir)
    
    # Write training loss
    for entry in data['train']:
        writer.add_scalar('1_train/loss', entry['loss'], entry['step'])
    
    # Write evaluation metrics
    for entry in data['eval']:
        writer.add_scalar('2_eval/micro_precision', entry['micro_precision'], entry['step'])
        writer.add_scalar('2_eval/micro_recall', entry['micro_recall'], entry['step'])
        writer.add_scalar('2_eval/micro_f1', entry['micro_f1'], entry['step'])
        writer.add_scalar('2_eval/macro_precision', entry['macro_precision'], entry['step'])
        writer.add_scalar('2_eval/macro_recall', entry['macro_recall'], entry['step'])
        writer.add_scalar('2_eval/macro_f1', entry['macro_f1'], entry['step'])
    
    writer.close()
    print(f"✅ Successfully wrote TensorBoard logs to {log_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', required=True, help='Path to training log file')
    parser.add_argument('--output-dir', default='./tensorboard_logs', help='Output directory')
    args = parser.parse_args()
    
    # Process the file
    data = parse_log_file(args.log_file)
    
    # Verify we found metrics
    if not data['eval']:
        print("⚠️ Warning: No evaluation metrics found in log file!")
        print("Check that your log contains lines like:")
        print("Micro P: 11.83%  Micro R: 34.51%  Micro F1: 17.62%")
        print("Macro P: 6.60%  Macro R: 15.76%  Macro F1: 8.05%")
    else:
        print(f"Found {len(data['eval'])} evaluation records")
    
    if not data['train']:
        print("⚠️ Warning: No training loss data found!")
    
    # Write the logs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_to_tensorboard(data, args.output_dir)
