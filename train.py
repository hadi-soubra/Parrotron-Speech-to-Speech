import os
import time
import random
import argparse
import datetime

# torch 관련
import torch
import torch.nn as nn
import torch.optim as optim

from models.build import build_model
from models.eval_distance import eval_wer
from models.data_loader import SpectrogramDataset, AudioDataLoader, EOS_token, index2char
from models.loss_function import ParrotronLoss, ParrotronLossNoASR

def compute_wer(preds, labels):
    total_wer = 0
    total_wer_len = 0

    for label, pred in zip(labels, preds):
        units = []
        units_pred = []
        for a in label:
            if a == EOS_token: # eos
                break
            units.append(index2char[a])

        for b in pred:
            if b == EOS_token: # eos
                break
            units_pred.append(index2char[b])

        label = ''.join(units)
        pred = ''.join(units_pred)

        total_wer += eval_wer(pred, label)
        total_wer_len += len(label.split())

    return total_wer, total_wer_len

def run_epoch(model, loader, criterion, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0
    total_wer = 0
    total_wer_len = 0

    start_time = time.time()
    total_batch_num = len(loader)
    with torch.set_grad_enabled(training):
        for i, data in enumerate(loader):
            seqs, targets, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths = data

            seqs = seqs.to(device) # (batch_size, time, freq)
            targets = targets.to(device) # (batch_size, max_len)
            tts_seqs = tts_seqs.to(device)

            # evaluation always uses teacher forcing so the loss is comparable between epochs
            tf = args.tf if training else 1.0
            mel_outputs_postnet, mel_outputs, gate_outputs, txt_outputs = model(seqs, tts_seqs, targets, tf)

            if args.no_asr:
                loss = criterion(mel_outputs_postnet, mel_outputs, gate_outputs, tts_seqs, tts_seq_lengths)
            else:
                loss = criterion(mel_outputs_postnet, mel_outputs, gate_outputs, txt_outputs, targets, tts_seqs, tts_seq_lengths)
            total_loss += loss.item()

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if not args.no_asr:
                hypothesis = txt_outputs.max(-1)[1]
                wer, wer_len = compute_wer(hypothesis.cpu().numpy(), targets.cpu().numpy())
                total_wer += wer
                total_wer_len += wer_len

            if training and i % 100 == 0:
                print('{} train_batch: {:4d}/{:4d}, train_loss: {:.4f}, train_time: {:.2f}'
                      .format(datetime.datetime.now(), i, total_batch_num, loss.item(), time.time() - start_time))
                start_time = time.time()

    final_wer = (total_wer / total_wer_len) * 100 if total_wer_len else float('nan')

    return total_loss / total_batch_num, final_wer

def main():
    parser = argparse.ArgumentParser(description='Train Parrotron')
    parser.add_argument('--train_csv', default='./label,csv/train.csv')
    parser.add_argument('--val_csv', default='./label,csv/test.csv')
    parser.add_argument('--save_dir', default='./plz_load')
    parser.add_argument('--no_asr', action='store_true', help='train without the auxiliary ASR decoder (E0)')
    parser.add_argument('--resume', default=None, help='checkpoint to continue training from')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tf', type=float, default=1.0, help='teacher forcing ratio')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123456)
    args = parser.parse_args()

    name = 'parrotron_no_asr' if args.no_asr else 'parrotron'
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, name + '.txt')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #-------------------------- Model Initialize --------------------------
    model = build_model(with_asr=not args.no_asr).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    #-------------------------- Loss Initialize --------------------------
    if args.no_asr:
        criterion = ParrotronLossNoASR()
    else:
        criterion = ParrotronLoss(nn.CrossEntropyLoss(ignore_index=0))

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.999),
                           eps=1e-06,
                           weight_decay=1e-06)

    #-------------------------- Data load --------------------------
    train_loader = AudioDataLoader(dataset=SpectrogramDataset(args.train_csv),
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size,
                                   drop_last=True)

    val_loader = AudioDataLoader(dataset=SpectrogramDataset(args.val_csv),
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size)

    print('{} 를 학습합니다. device = {}'.format(name, device))

    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_time = time.time()
        train_loss, train_wer = run_epoch(model, train_loader, criterion, device, args, optimizer)
        train_total_time = time.time() - train_time

        eval_time = time.time()
        eval_loss, test_wer = run_epoch(model, val_loader, criterion, device, args)
        eval_total_time = time.time() - eval_time

        lines = ['Epoch %d (Train) Loss %0.4f WER %0.2f time %0.1f' % (epoch+1, train_loss, train_wer, train_total_time),
                 'Epoch %d (Eval) Loss %0.4f WER %0.2f time %0.1f' % (epoch+1, eval_loss, test_wer, eval_total_time)]
        print('\n'.join(lines))
        with open(log_path, 'a') as f:
            f.write('\n'.join(lines) + '\n')

        if eval_loss < best_loss:
            print("best model을 저장하였습니다.")
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_' + name + '.pth'))
            best_loss = eval_loss

        torch.save(model.state_dict(), os.path.join(args.save_dir, name + '.pth'))

if __name__ == '__main__':
    main()
