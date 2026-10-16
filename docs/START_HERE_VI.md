# Bắt đầu trên Mac — Robot-as-Code

## Hệ thống này làm được gì

Một robot giúp việc hoàn toàn bằng phần mềm, chạy trong một căn hộ 3D có vật lý thật (MuJoCo):

- **Cơ thể robot** khai báo trong `robots/helper_rover.yaml`: khung, pin 150 Wh, 2 bánh chủ động có giới hạn mô-men, 2 bánh bi, trục nâng, tay vươn, cổ tay trượt ngang, kẹp 2 ngón có cảm biến lực, camera RGB-D, LiDAR 270°, IMU, encoder, bumper. File MuJoCo được **sinh tự động** từ YAML.
- **Ngôi nhà** (`worlds/apartment/`): phòng khách, bếp, phòng làm việc, nội thất và người dùng ngồi trên sofa.
- **Phần mềm onboard** chỉ nhìn thế giới qua cảm biến (không đọc vị trí thật từ simulator):
  - định vị bằng encoder + gyro + so khớp LiDAR với bản đồ;
  - điều hướng: lập đường, tránh vật cản thấy bằng LiDAR, lập lại đường khi bị chặn;
  - nhận thức: tìm vật trong ảnh RGB-D, phân biệt chai / cốc / lon / táo theo màu và hình dạng;
  - gắp bằng **ma sát thật**, kiểm tra bằng cảm biến lực ngón và độ mở kẹp;
  - an toàn: bumper, độ nghiêng, pin.
- **Não AI**: hiểu lệnh tiếng Việt/Anh (có dấu hoặc không dấu) → kế hoạch gồm các skill → thực thi, tự kiểm chứng, tự phục hồi khi lỗi, báo cáo trung thực.
- **Trọng tài**: chấm điểm bằng trạng thái thật của simulator (vật có thật sự nằm đúng chỗ không, có va chạm không, robot có “nói dối” là xong không).

## Cài đặt một lần

```bash
cd robot-as-code-mac-mvp
bash scripts/bootstrap_mac.sh
```

Script tạo `.venv`, cài MuJoCo + numpy + scipy + pyyaml, chạy toàn bộ test và một nhiệm vụ đầy đủ không giao diện. Cuối cùng in `READY`.

## Chạy

```bash
source .venv/bin/activate

# Mở nhà 3D, gõ nhiệm vụ liên tục (macOS bắt buộc dùng mjpython cho cửa sổ 3D)
mjpython -m robotcode.cli --interactive

# Một nhiệm vụ, có cửa sổ 3D
mjpython -m robotcode.cli --task "Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao."

# Không cửa sổ (nhanh, in nhật ký + kết quả trọng tài)
python -m robotcode.cli --headless --task "Tìm chai nước màu đỏ trong phòng bếp rồi mang đến bàn làm việc."

# Chỉ xem kế hoạch, không chạy vật lý
python -m robotcode.cli --dry-run --task "Mang cái cốc xanh dương tới đích"
```

Trong cửa sổ 3D: chấm xanh dương là đường robot định đi, vòng vàng là vật robot đang nhớ, mũi tên xanh lá là vị trí robot *tin* là mình đang đứng. Bỏ `--fast` để xem đúng tốc độ thật.

Ví dụ lệnh robot hiểu:

| Lệnh | Kế hoạch |
|---|---|
| Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao. | đi phòng khách → tìm chai trên bàn → gắp → đặt cạnh người dùng |
| Find the red bottle in the kitchen and bring it to my desk. | tìm chai đỏ (ưu tiên bếp) → gắp → đặt lên bàn làm việc |
| Mang cái cốc xanh dương từ bếp lên bàn làm việc | tìm cốc xanh trong bếp → gắp → đặt lên bàn làm việc |
| Mang chai tới đích | tìm chai → gắp → đặt vào vùng xanh trên sàn |
| Đi vào bếp / Về trạm sạc / Pin còn bao nhiêu? / Dừng lại! / Nhìn xung quanh | điều hướng / trạng thái / dừng / quan sát |

Lệnh không làm được (ví dụ “dọn cả nhà”) sẽ bị từ chối kèm lý do, không bịa kế hoạch.

## Kiểm thử và đo lường

```bash
python -m unittest discover -s tests -v        # 28 test: đơn vị, vật lý, nhiệm vụ đầy đủ
python -m robotcode.evaluate --episodes 20     # mỗi nhiệm vụ 20 lần với vị trí ngẫu nhiên
```

Bảng kết quả có: tỉ lệ thành công, số lần robot báo thành công sai (`false+`), va chạm với nhà, va chạm với đồ vật, quãng đường, sai số định vị. Chi tiết từng lượt nằm trong `artifacts/eval_*.json`. Số liệu mới nhất: xem README.

## Mọi thứ là code / cấu hình

| Muốn thay đổi | Sửa file |
|---|---|
| Kích thước, khối lượng, motor, cảm biến, nhiễu cảm biến của robot | `robots/helper_rover.yaml` (hoặc tạo robot mới, ví dụ `robots/scout.yaml` — robot không tay) |
| Nhà, nội thất | `worlds/apartment/world.xml` |
| Tên phòng, bàn, địa điểm, từ đồng nghĩa tiếng Việt/Anh | `worlds/apartment/semantic.yaml` |
| Loại đồ vật, màu, khối lượng | `objects/catalog.yaml` |
| Nhiệm vụ, vị trí đồ vật (có ngẫu nhiên), tiêu chí thành công | `tasks/*.yaml` |

Mỗi lượt chạy được xác định hoàn toàn bởi (file nhiệm vụ, seed). File MuJoCo sinh ra nằm ở `build/scenes/`, bản đồ ở `build/maps/` (có ảnh `.pgm` kiểu Nav2).

## Bật LLM (tuỳ chọn)

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY="..."            # không commit key lên git
mjpython -m robotcode.cli --planner llm --task "Lấy giúp tôi cái cốc màu xanh trong bếp nhé"
```

LLM chỉ được viết **kế hoạch gồm các skill có sẵn**; kế hoạch được kiểm tra (skill có tồn tại không, địa điểm/đồ vật có trên bản đồ/catalog không, robot có đủ phần cứng không) trước khi robot cử động. LLM không bao giờ điều khiển motor trực tiếp. Dùng API có thể phát sinh chi phí.

## Giới hạn hiện tại

- Nhận diện theo màu + hình dạng (chưa có YOLO/VLM): đồ vật phải có màu trong catalog.
- Tay kiểu Cartesian + kẹp song song: gắp đồ đứng trên bàn/quầy; chưa gắp dưới sàn, chưa có tay 6 bậc.
- Bản đồ lấy từ hình học ngôi nhà (như bản vẽ mặt bằng), chưa tự lập bản đồ bằng SLAM.
- Chưa có ROS 2 / robot thật; ranh giới để chuyển sang phần cứng là `hal.py` (bảng ánh xạ ROS 2 trong `docs/ARCHITECTURE.md`).
