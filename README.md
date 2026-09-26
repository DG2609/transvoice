# TransVoice

Dịch giọng nói **Nhật ↔ Anh ↔ Việt** ngay trên máy, **chỉ dùng CPU** (không cần GPU, không cần internet).
Âm thanh cuộc gọi/video được nhận dạng và dịch, hiện thành phụ đề nổi trên mọi cửa sổ. Không có gì rời khỏi máy.

Giai đoạn 1 chỉ dịch ra **chữ**. Đọc bản dịch thành giọng nói (TTS) để giai đoạn sau, xem [docs/DECISIONS.md](docs/DECISIONS.md).

## Chạy

```bat
transvoice.bat                         :: dịch âm thanh máy (Zoom/Teams/YouTube...) sang tiếng Việt, tự nhận ngôn ngữ
transvoice.bat --them ja               :: cố định: người kia nói tiếng Nhật (nhanh và chắc hơn tự nhận)
transvoice.bat --me ja                 :: bạn đọc tiếng Nhật: dịch mọi thứ sang tiếng Nhật
transvoice.bat --sources system,mic    :: dịch cả giọng bạn (micro) sang ngôn ngữ của họ
transvoice.bat --glossary glossary.json:: dùng từ điển thuật ngữ riêng (tên công ty, sản phẩm...)
transvoice.bat --file cuochop.wav      :: dịch một file ghi âm, in kết quả ra màn hình
transvoice.bat --list-devices          :: xem danh sách micro / loa
transvoice.bat --system-device 28      :: nghe loa/tai nghe cụ thể (khi Zoom không phát ra loa mặc định)
```

Phụ đề chạy theo lời nói: mỗi 2–5 giây, bản dịch của **cả câu đang nói** được cập nhật (chữ xám). Khi người nói
dứt câu, bản dịch được chốt (chữ trắng) và lưu lại. Nhờ luôn dịch cả câu, ngữ cảnh không bị mất như kiểu
"nghe tới đâu dịch tới đó". Máy yếu có thể dùng `--no-drafts` để chỉ dịch khi hết câu.

Cửa sổ phụ đề:

- Kéo để di chuyển, chuột phải để mở menu (cỡ chữ, khoá, thoát).
- **Ctrl+Alt+L**: khoá / mở khoá. Khi khoá, chuột bấm xuyên qua phụ đề xuống ứng dụng bên dưới.
- **Ctrl+Alt+Q**: thoát.

Mỗi phiên được lưu trong `sessions/` (`.md` để đọc, `.jsonl` để xử lý).

## Từ điển thuật ngữ

Sao chép `glossary.example.json` rồi sửa. Mỗi mục là một thuật ngữ viết bằng các ngôn ngữ:

```json
{"terms": [{"vi": "báo giá", "ja": "見積もり", "en": "quotation"}]}
```

Thứ trong tuần tiếng Việt ("thứ năm", "thứ 6", "chủ nhật") đã được xử lý tự động.

## Cài đặt từ bản build (Releases)

Tải gói cho máy của bạn ở trang **Releases**, giải nén, rồi tải model một lần (khoảng 2,9 GB, cần internet lúc này):

| Hệ điều hành | Gói | Tải model | Chạy |
|---|---|---|---|
| Windows 10/11 x64 | `TransVoice-windows-x64.zip` | `TransVoice.exe --download-models` | `TransVoice.bat` |
| macOS Apple Silicon | `TransVoice-macos-arm64.tar.gz` | `./TransVoice --download-models` | `./run.sh` |
| Linux x64 | `TransVoice-linux-x64.tar.gz` | `./TransVoice --download-models` | `./run.sh` |

Máy không có internet: chép nguyên thư mục `models/` từ một máy đã tải sang, đặt cạnh file chạy.
Khi chạy, app dùng khoảng 3,5–4 GB RAM.

**Âm thanh máy trên macOS / Linux:**

- **macOS** không cho ứng dụng thu trực tiếp âm thanh máy. Cài **BlackHole 2ch** (miễn phí), trong *Audio MIDI Setup*
  tạo *Multi-Output Device* gồm loa + BlackHole và chọn nó làm đầu ra. TransVoice tự tìm thấy BlackHole.
  Bản build chưa được ký: lần đầu hãy chuột phải → *Open*, hoặc chạy `xattr -dr com.apple.quarantine TransVoice`.
- **Linux** (glibc 2.27+, ví dụ Ubuntu 18.04 trở lên) dùng PulseAudio/PipeWire. Cần các gói
  `pulseaudio-utils libportaudio2 libgomp1`, ví dụ `sudo apt install pulseaudio-utils libportaudio2 libgomp1`.

## Cài đặt từ mã nguồn

Cần Python 3.12.

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m transvoice --download-models
.venv\Scripts\python -m transvoice
```

Build gói cho hệ điều hành đang dùng: `python packaging/build.py` (chạy test trước, kết quả ở `dist/`).
Đẩy một tag `v*` lên GitHub thì GitHub Actions tự build cả Windows, macOS và Linux rồi đính kèm vào Release.

## Cấu hình và hiệu năng

| Khâu | Model |
|---|---|
| Nhận dạng tiếng Nhật | parakeet-tdt_ctc-0.6b-ja |
| Nhận dạng tiếng Anh | SenseVoice-Small |
| Nhận dạng tiếng Việt | Zipformer-30M-vi |
| Nhận diện ngôn ngữ | Whisper-base |
| Cắt câu | Silero VAD, có tự cân bằng âm lượng |
| Dịch | HY-MT1.5-1.8B (llama.cpp) + từ điển thuật ngữ |

Kết quả đo chi tiết nằm trong [results/REPORT.md](results/REPORT.md). Mã đo nằm trong `bench/` và `scripts/`, hướng dẫn chạy ở [bench/README.md](bench/README.md).
