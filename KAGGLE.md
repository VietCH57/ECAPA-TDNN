# Chạy baseline ECAPA-TDNN cho Vi-SRED26 trên Kaggle

Kiến trúc mô hình (`model.py`) và loss (`loss.py`) **giữ nguyên** so với repo gốc. Mọi hyperparameter giờ đều chỉnh được bằng dòng lệnh hoặc file YAML. Giá trị mặc định giống hệt code gốc.

## 0. Chuẩn bị (một ô trong notebook)

```python
!git clone https://github.com/<you>/ECAPA-TDNN.git /kaggle/working/ECAPA-TDNN
%cd /kaggle/working/ECAPA-TDNN
# Kaggle đã có sẵn torch/torchaudio/numpy/scipy/sklearn/soundfile/yaml -> KHÔNG chạy pip install -r requirements.txt
```

Các dataset cần add vào notebook:

| Dataset | Đường dẫn |
|---|---|
| Train | `/kaggle/input/datasets/hongvitchugo/vi-sred26-train` (`Vi-SRED26-Train/data`, `train_split.txt`, `val_split.txt`, `val_trials.txt`) |
| Public test | `/kaggle/input/datasets/hongvitchugo/vi-sred26-publictest` (`data/`, `public-test.txt`) |
| MUSAN | `/kaggle/input/datasets/nhattruongdev/musan-noise` (code tự tìm `musan/{music,noise,speech}`) |
| RIR | `/kaggle/input/datasets/nhattruongdev/rirs-noises` (code tự tìm `.../simulated_rirs`) |

## 1. Train

```bash
!python train.py --config configs/vi_sred26_baseline.yaml
```

Ghi đè bất kỳ tham số nào ngay trên dòng lệnh, ví dụ:

```bash
!python train.py --config configs/vi_sred26_baseline.yaml --batch_size 200 --max_epoch 60 --save_path /kaggle/working/exps/bs200
```

Khi mở đầu, log sẽ in ra thư mục MUSAN/RIR tìm được, số file trong mỗi loại và xác suất của từng loại augment. **Hãy kiểm tra các dòng này trước khi để chạy lâu.**

Đầu ra nằm trong `save_path`:

| File | Nội dung |
|---|---|
| `config.json` | Toàn bộ tham số của lần chạy |
| `speakers.txt` | Speaker → chỉ số lớp |
| `score.txt` | Mỗi lần đánh giá ghi 1 dòng: LR, loss, ACC, EER, bestEER, minDCF |
| `model/model_XXXX.model` | Trọng số, cùng định dạng repo gốc |
| `model/best.model` | Trọng số có EER val thấp nhất |
| `model/last.ckpt` | Trọng số + optimizer + epoch, dùng để resume chính xác |

**Resume khi session Kaggle bị ngắt:** chạy lại đúng lệnh cũ với cùng `--save_path`. Code sẽ tự đọc `model/last.ckpt` và train tiếp từ epoch kế tiếp. Nếu checkpoint nằm ở output của một notebook trước (đã được add làm dataset), có hai cách:
- copy thư mục đó vào `/kaggle/working/exps/...`, hoặc
- dùng `--resume /kaggle/input/<...>/model/last.ckpt`.

**Fine-tune từ mô hình VoxCeleb của tác giả** (tuỳ chọn): thêm `--initial_model path/to/pretrain.model`. Lớp head 5994 speaker bị lệch kích thước sẽ tự bị bỏ qua.

## 2. Đánh giá

```bash
# trên val (có nhãn)
!python infer.py --mode eval --model /kaggle/working/exps/baseline/model/best.model \
    --trials /kaggle/input/datasets/hongvitchugo/vi-sred26-train/val_trials.txt \
    --data_path /kaggle/input/datasets/hongvitchugo/vi-sred26-train/Vi-SRED26-Train/data \
    --out /kaggle/working/eval/val.txt
```

Nếu có `val_trials_meta.csv` (sinh từ notebook chia tập), thêm `--eval_meta <đường dẫn>` để in thêm EER_H, EER_X và EER theo độ dài. Cờ này cũng dùng được trong `train.py`.

## 3. Tạo file nộp cho public test

```bash
!python infer.py --mode score --model /kaggle/working/exps/baseline/model/best.model \
    --trials /kaggle/input/datasets/hongvitchugo/vi-sred26-publictest/public-test.txt \
    --data_path /kaggle/input/datasets/hongvitchugo/vi-sred26-publictest/data \
    --out /kaggle/working/submission/results.txt --zip
```

- `results.txt` có mỗi dòng dạng `file1.wav file2.wav score`, ngăn cách bằng đúng một dấu cách, **cùng thứ tự** với `public-test.txt`. Script tự kiểm tra điều này trước khi ghi.
- `results.zip` chứa `results.txt`, là file để nộp.
- Vì `public-test.txt` có nhãn (`target`/`nontarget`), script in luôn EER/minDCF trên public test.
- Thêm `--emb_cache /kaggle/working/submission/pub_emb.npz` để lưu embedding, lần sau chấm lại sẽ không phải tính lại.

## 4. Các tham số chính (`python train.py -h` để xem đủ)

| Nhóm | Tham số (mặc định = code gốc) |
|---|---|
| Dữ liệu | `--train_list --train_path --num_frames 200 --batch_size 400 --n_cpu 4 --seed -1` |
| Augment | `--musan_path --rir_path` (để `""` là tắt), `--aug_probs 1,1,1,1,1,1` (clean, reverb, babble, music, noise, tv), `--snr_noise 0,15 --snr_speech 13,20 --snr_music 5,15`, `--num_noise 1,1 --num_speech 3,8 --num_music 1,1` |
| SpecAugment | `--specaug true --specaug_freq 0,8 --specaug_time 0,10` |
| Tối ưu | `--max_epoch 80 --lr 0.001 --lr_decay 0.97 --lr_step` (0 = theo `test_step`, như gốc) `--weight_decay 2e-5 --grad_clip 0` |
| Nhiều GPU | `--data_parallel false` (true = chia batch train cho mọi GPU qua `DataParallel`) |
| Mô hình / loss | `--C 1024 --m 0.2 --s 30 --n_class` (0 = tự đếm từ train list) |
| Đánh giá | `--eval_list --eval_path --eval_meta --test_step 1 --eval_frames 300 --num_eval 5 --eval_mode both --dcf_ptarget 0.05` |
| Checkpoint | `--save_path --initial_model --resume auto --keep_last 0 --save_best true --log_interval 1 --device cuda --max_hours 0` |

Config `configs/vi_sred26_baseline.yaml` giữ nguyên các hyperparameter huấn luyện của bản gốc. File này chỉ đổi những thứ không ảnh hưởng tới việc học: đường dẫn, `test_step: 2` (đánh giá mỗi 2 epoch để tiết kiệm thời gian, còn LR vẫn giảm mỗi epoch nhờ `lr_step: 1`), `keep_last: 3`, `log_interval: 20`, `seed: 42`.

## 5. Khác biệt so với repo gốc

- **Sửa lỗi:**
  - `numpy.float` đã bị xoá trong NumPy ≥ 1.24, khiến code crash khi eval và khi dùng RIR.
  - Thiếu MUSAN hoặc RIR làm code crash; giờ loại augment tương ứng chỉ bị tắt.
  - `--n_class` mặc định 5994 (không đúng với tập này); giờ tự đếm từ train list.
- **LR decay tách khỏi `test_step`:** cờ `--lr_step`. Với giá trị mặc định, lịch LR giống hệt bản gốc.
- **Đọc nhiễu MUSAN theo đoạn:** chỉ đọc phần cần dùng, nhanh hơn với bản MUSAN gốc có file dài. Phân bố đoạn được chọn không đổi.
- **Eval được vector hoá:** EER và minDCF trùng khớp với code gốc, điểm từng cặp lệch tối đa khoảng 1e-7 do làm tròn float32.
- **Script cũ:** `trainECAPAModel.py` được giữ nguyên và vẫn chạy được.
