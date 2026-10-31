# NHAN DIEN CU CHI TAY - PYCHARM / PYTHON 3.12.2 (WINDOWS 64-BIT)

Bo ma nguon **viet lai de hoc tap**, ke thua phuong phap MediaPipe Holistic + LSTM tu:
https://github.com/ApheliosQ/sign-language-detection-using-lstm

Day la phien ban cho **Python 3.12.2**, khong can cai Python 3.10 va khong can su dung TensorFlow 2.15.1.

**Luu y:** Project chua kem mo hinh da huan luyen hoac du lieu cu chi. Can thu thap du lieu, huan luyen roi moi chay demo. Phan mem chi nhan dien 5 nhan da hoc, khong phai bo dich ngon ngu ky hieu day du.

## 1. Mo project trong PyCharm

1. Giai nen ZIP. Vao **File > Open**, chon thu muc `NhanDienCuChiTay_Python312` chua file `config.py`.
2. Vao **Settings > Project > Python Interpreter**. Neu PyCharm dang su dung `.venv` Python 3.12.2 trong project **cu**, chon lai interpreter cho dung thu muc moi. Nen tao **.venv moi ben trong project moi** de tach biet thu vien cu.
3. Neu can tao moi bang Terminal cua PyCharm (PowerShell):

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python --version
   ```

   Neu PowerShell bao loi khong cho chay Activate.ps1, co the dung PowerShell khong kich hoat:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe Kiem_Tra_Moi_Truong.py
   ```

4. Cai thu vien (trong Terminal co `(.venv)` va `Python 3.12.2`):

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip check
   python Kiem_Tra_Moi_Truong.py
   ```

5. Thanh cong khi in `[OK]` cho thu vien, `MediaPipe Holistic legacy: OK`, `Keras LSTM build: OK`.

**Neu muon giu dung `.venv` cua project cu:** Khong can go Python. Chep code moi vao project cu (sao luu file truoc); dat lai `requirements.txt` va chay `python -m pip install -r requirements.txt`. Nhung de tranh xung dot, khuyen nghi dung project moi va `.venv` moi.

## 2. Thu tu chay

| Buoc | File | Ket qua |
|---|---|---|
| 1 | `Kiem_Tra_Moi_Truong.py` | Kiem tra Python, OpenCV, MediaPipe, TensorFlow, Keras |
| 2 | `Thu_Thap_Data.py` | Mo webcam va luu 30 frames/chuoi vao MP_Data/ |
| 3 | `Kiem_Tra_Du_Lieu.py` | Dem chuoi hop le theo nhan/nguoi |
| 4 | `huan_luyen_mo_hinh.py` | Train LSTM, luu mo hinh `.keras` va bieu do |
| 5 | `Run_No_Sound.py` | Demo webcam, hien thi tieng Viet |
| 6 | `Run_With_Sound.py` | Demo webcam va phat MP3 |

PyCharm: mo file, click phai > Run. Trong Terminal co the chay `python Thu_Thap_Data.py` v.v.

## 3. Thu thap du lieu

- Trong `config.py`: 5 nhan `khong_cu_chi`, `xin_chao`, `cam_on`, `xin_loi`, `tam_biet`.
- `SEQUENCES_PER_ACTION = 40` so chuoi moi nhan trong mot lan chay; co the doi thanh 10 de test quy trinh.
- Chay `Thu_Thap_Data.py`, nhap `p01` (ma nguoi) va `s01` (ma buoi quay).
- Khi hien `CHUAN BI`, chuan bi cu chi; khi `DANG QUAY`, lam cu chi trong 30 khung hinh.
- Chay lai voi `p02`, `p03`, ... de tang da dang. Nen co tu 4 nguoi tro len (ly tuong co nhieu hon) va quay trong nhieu dieu kien.
- Nhan **Q** trong cua so webcam de thoat, chuoi chua du 30 frame bi xoa.
- `khong_cu_chi` la cac trang thai khong thuoc nhom cu chi can nhan dien.
- Thong tin ma nguoi phuc vu chia tap, khong can luu ten that. Xin phep truoc khi ghi hinh.

**Luu y ve nhan:** Nhung nhan la nhom cu chi do nguoi thu thap quy uoc. Khong duoc khang dinh la ngon ngu ky hieu Viet Nam tieu chuan neu chua doi chieu voi nguon uy tin.

## 4. Huan luyen

- Chay `Kiem_Tra_Du_Lieu.py` de xem so chuoi hop le.
- Chay `huan_luyen_mo_hinh.py`. Doi so khi train: `(N, 30, 1662)`.
- Model: LSTM(64) -> Dropout -> LSTM(64) -> Dense(64) -> Dropout -> Dense(5, softmax).
- Tinh nang: Adam 0.001, batch 16, 100 epoch toi da, EarlyStopping + ModelCheckpoint.
- Neu >=4 nguoi: split theo nguoi, neu <4 nguoi: split theo chuoi va can canh bao ve do tin cay.
- Mo hinh: `models/gesture_model.keras`; thong tin: `models/model_info.json`.
- Bao cao: `results/accuracy.png`, `results/loss.png`, `results/confusion_matrix.png`, `results/classification_report.txt`.

## 5. Chay demo

- Chay `Run_No_Sound.py` de demo truoc.
- Sau khi webcam on dinh, chay `Run_With_Sound.py` neu muon co am thanh MP3.
- Nhan **Q** trong webcam de thoat. Cac nhan tieng Viet duoc ve bang Pillow de co dau.
- Ket qua sai? Tang so nguoi duoc quay, can bang mau giua cac nhan, cai thien anh sang, lam dung cu chi.
- Viec du doan cac nhan thuong gap xuyen suot chuoi co the hien thi tre; co the thay `STABLE_SECONDS` trong config.

## 6. Loi thuong gap

- `tensorflow==2.15.1 ... No matching distribution`: Ban dang cai bo yeu cau cu. Kiem tra `requirements.txt` phai la `tensorflow==2.16.2`.
- `No module named cv2`: Lenh pip install bi dung giua chung hoac sai interpreter. Chay `python -m pip install -r requirements.txt`.
- `mediapipe has no attribute solutions`: Chua dung `mediapipe==0.10.21` hoac dang dung interpreter sai.
- `ModuleNotFoundError` trong PyCharm khi Terminal OK: PyCharm Run Configuration dang chon interpreter khac; chinh o Settings.
- `Cannot open camera`: Tat Zoom/Teams, cap quyen camera, thu `CAMERA_INDEX = 1`.
- `Chua co model`: Thu thap du lieu va chay huan luyen truoc.
- `DLL load failed` khi import TensorFlow tren Windows: Cai/repair Microsoft Visual C++ Redistributable x64 tu Microsoft va khoi dong lai PyCharm.
- Chay CPU se cham hon GPU; tren Windows native TensorFlow 2.16 dung CPU; day la tinh huong binh thuong.

## 7. Ghi chu ket qua va nguon

- Bo code nay la ban tham khao de nhom tu train va test. Khong co ket qua do chinh xac duoc dam bao truoc.
- Neu muon so sanh voi repository goc, ghi ro diem ke thua (MediaPipe + LSTM 30 frames 1662 features) va diem tu them (5 nhan, tach file, ket qua danh gia, cach chia du lieu).
- Chua thuc thi duoc huan luyen TensorFlow tren Windows/webcam thuc te trong moi truong tao ZIP; da kiem tra tinh hop le cua code va tep nen.
