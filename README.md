# ECAPA-TDNN (bản adapt cho Vi-SRED26)

Repo này được adapt từ [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN), một bản tái hiện không chính thức của ECAPA-TDNN cho bài toán speaker verification. README gốc của tác giả được giữ nguyên tại [README_original.md](README_original.md).

**Những gì đã thay đổi:** kiến trúc mô hình (`model.py`) và loss (`loss.py`) được giữ nguyên. Mọi hyperparameter của train và inference giờ được truyền qua dòng lệnh hoặc file YAML, với giá trị mặc định giống hệt code gốc. Repo có thêm `infer.py` để chấm điểm, đánh giá và trích embedding, đồng thời sửa các lỗi tương thích với NumPy/PyTorch mới.

Chi tiết các thay đổi và hướng dẫn riêng cho Kaggle nằm trong [KAGGLE.md](KAGGLE.md).

---

## Cài đặt

```bash
git clone https://github.com/<you>/ECAPA-TDNN.git
cd ECAPA-TDNN
pip install -r requirements.txt   # trên Kaggle/Colab: bỏ qua bước này, các thư viện đã có sẵn
```

## Định dạng dữ liệu

| File | Định dạng mỗi dòng | Ví dụ |
|---|---|---|
| Train list (`--train_list`) | `<speaker> <đường/dẫn.wav>` (cách bằng dấu cách hoặc tab) | `id00000	id00000/0009.wav` |
| Trial list (`--eval_list`, `--trials`) | `<label> <a.wav> <b.wav>` hoặc `<a.wav> <b.wav> target\|nontarget` hoặc `<a.wav> <b.wav>` | `1 id01549/0277.wav id01549/0099.wav` |

Đường dẫn trong các list là tương đối so với `--train_path`, `--eval_path` hoặc `--data_path`. Audio cần ở dạng WAV 16 kHz mono.

Augment bằng MUSAN và RIR là tuỳ chọn:
- `--musan_path` trỏ tới thư mục chứa `music/ noise/ speech/`, hoặc bất kỳ thư mục cha nào của nó. Dùng được cả bản MUSAN gốc lẫn bản `musan_split`.
- `--rir_path` trỏ tới `RIRS_NOISES` hoặc thư mục cha của nó. Code tự tìm `simulated_rirs` bên trong.
- Để trống một trong hai thì loại augment tương ứng sẽ bị tắt.

## Train

```bash
# toàn bộ tham số qua dòng lệnh
python train.py --train_list data/train_split.txt --train_path data/wav \
                --eval_list data/val_trials.txt --eval_path data/wav \
                --musan_path /data/musan --rir_path /data/RIRS_NOISES \
                --save_path exps/exp1

# hoặc dùng file config, rồi ghi đè vài tham số trên dòng lệnh
python train.py --config configs/vi_sred26_baseline.yaml --batch_size 200

# xem toàn bộ tham số và giá trị mặc định
python train.py -h
```

| Nhóm | Tham số chính |
|---|---|
| Dữ liệu | `--num_frames --batch_size --n_cpu --seed` |
| Augment | `--aug_probs` (clean, reverb, babble, music, noise, tv) `--snr_noise --snr_speech --snr_music --num_noise --num_speech --num_music --specaug --specaug_freq --specaug_time` |
| Tối ưu | `--max_epoch --lr --lr_decay --lr_step --weight_decay --grad_clip` |
| Mô hình / loss | `--C --m --s --n_class` (mặc định tự đếm từ train list) |
| Đánh giá | `--test_step --eval_frames --num_eval --eval_mode --dcf_ptarget --eval_meta` |
| Checkpoint | `--save_path --initial_model --resume --keep_last --save_best --log_interval --device` |

Kết quả được lưu trong `--save_path`:
- `score.txt`: loss, độ chính xác, EER và minDCF qua từng lần đánh giá;
- `config.json`: toàn bộ tham số của lần chạy;
- `model/`: `model_XXXX.model`, `best.model` (EER thấp nhất) và `last.ckpt`.

**Resume:** chạy lại đúng lệnh cũ, code sẽ tự train tiếp từ `model/last.ckpt`. Cũng có thể chỉ định checkpoint bằng `--resume path/to/last.ckpt`.

**Fine-tune:** thêm `--initial_model path/to/model`. Lớp phân loại bị lệch kích thước sẽ tự bị bỏ qua.

**Chỉ đánh giá:** `python train.py --eval --initial_model exps/exp1/model/best.model --eval_list ... --eval_path ...`

## Inference

```bash
# chấm điểm một trial list -> "file1 file2 score" theo đúng thứ tự, kèm results.zip để nộp
python infer.py --mode score --model exps/exp1/model/best.model \
                --trials test/public-test.txt --data_path test/data \
                --out submission/results.txt --zip

# đánh giá trên trial list có nhãn (EER, minDCF)
python infer.py --mode eval --model exps/exp1/model/best.model \
                --trials data/val_trials.txt --data_path data/wav

# trích embedding (192 chiều) -> .npz
python infer.py --mode embed --model exps/exp1/model/best.model \
                --file_list data/val_split.txt --data_path data/wav --out emb.npz
```

Một số cờ hay dùng:
- `--emb_cache file.npz`: lưu embedding để lần sau chấm lại không phải tính lại.
- `--eval_frames`, `--num_eval`, `--eval_mode`: đổi cách lấy embedding. Mặc định là cả câu kết hợp 5 đoạn 3 giây, như bản gốc.
- `--C`: phải trùng với giá trị đã dùng khi train.

Khi chế độ `score` nhận trial list có nhãn, script in thêm EER và minDCF.

## Cấu trúc repo

| File | Vai trò |
|---|---|
| `model.py`, `loss.py` | ECAPA-TDNN và AAM-softmax, giữ nguyên bản gốc |
| `train.py`, `config.py` | Train qua dòng lệnh hoặc YAML |
| `infer.py`, `inference.py` | Chấm điểm, đánh giá, trích embedding |
| `ECAPAModel.py`, `dataLoader.py`, `tools.py` | Vòng train/eval, data loader + augment, EER/minDCF (code gốc, đã tham số hoá) |
| `trainECAPAModel.py` | Script train gốc, vẫn dùng được |
| `configs/vi_sred26_baseline.yaml` | Config baseline cho Vi-SRED26 trên Kaggle |
| `KAGGLE.md` | Hướng dẫn chạy trên Kaggle và danh sách thay đổi chi tiết |

## Giấy phép

MIT, theo repo gốc. Xem [LICENSE.md](LICENSE.md).

## Reference

Repo gốc
```
https://github.com/TaoRuijie/ECAPA-TDNN
```

ECAPA-TDNN
```
@inproceedings{desplanques2020ecapa,
  title={{ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification}},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech 2020},
  pages={3830--3834},
  year={2020}
}
```

Báo cáo tái hiện của tác giả repo gốc
```
@article{das2021hlt,
  title={HLT-NUS SUBMISSION FOR 2020 NIST Conversational Telephone Speech SRE},
  author={Das, Rohan Kumar and Tao, Ruijie and Li, Haizhou},
  journal={arXiv preprint arXiv:2111.06671},
  year={2021}
}
```

VoxCeleb trainer
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Interspeech},
  year={2020}
}
```
