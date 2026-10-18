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

## Robot nhện (gập thành logo)

`robots/spider_bot.yaml` là robot thứ hai: nhện 8 chân, mỗi chân 3 servo (24 servo), nặng 1.4 kg, có IMU, cảm biến chạm ở 8 bàn chân, LiDAR mini và camera. Nó chạy **cùng phần mềm** với robot giúp việc: định vị, điều hướng, nhận diện, planner đều dùng lại; chỉ thêm IK chân và bộ tạo dáng đi (tetrapod xen kẽ, như nhện thật).

```bash
# Đi vào phòng làm việc rồi gập lại thành logo
mjpython -m robotcode.cli --scenario spider_fold_in_office

# Tìm quả táo lăn xuống gầm bàn bếp (nhện cao 12 cm nên chui được gầm bàn)
python -m robotcode.cli --headless --scenario spider_find_under_table

# Điều khiển nhện bằng lời: "gập lại", "biến thành logo", "mở ra rồi đi vào bếp", "đứng dậy"
mjpython -m robotcode.cli --scenario spider_fold_in_office --interactive
```

Khi gập: thân hạ xuống nằm trên sàn, rồi từng nhóm 4 chân nhấc lên và thu vào, chân trước quét về trước, chân sau quét ra sau, tạo dáng logo con nhện. Đường gập được kiểm tra trước để các chân không va vào nhau. Lệnh đi lại khi đang gập sẽ tự mở ra trước. Planner phân biệt **"gập"** (gấp chân) với **"gắp"** (gắp đồ), kể cả khi gõ không dấu thì vẫn ưu tiên đúng ngữ cảnh.

## Robot hình người đấu boxing

`robots/humanoid_boxer.yaml`: robot hai chân cao 1,3 m, 47 kg, 21 servo (6 mỗi chân, xoay eo, 4 mỗi tay), găng có đệm, IMU, encoder, cảm biến lực gót/mũi chân, cảm biến va chạm ở đầu/thân/găng, camera RGB-D ở đầu. Hai robot (góc đỏ, góc xanh) đấu trên võ đài, có trọng tài chấm điểm.

```bash
mjpython -m robotcode.match                          # xem trận đấu 3D, có bảng điểm
python -m robotcode.match --headless --rounds 1      # không cửa sổ, in kết quả
python -m robotcode.match --bench 12 --rounds 1      # 12 trận song song + thống kê
python -m robotcode.match --rounds 1 --video build/video/tran_dau.mp4   # quay video (không cần màn hình)
```

Mỗi võ sĩ tự **giữ thăng bằng** (capture point, ước lượng trọng tâm từ IMU + động học chân, tự hiệu chỉnh sai số IMU lúc khởi động), **di chuyển chân** kiểu boxing (bước tới – kéo chân sau, xoay người), **tìm đối thủ bằng camera** (sai số đầu ~2 cm), **ra đòn** jab / cross / hook / đấm thân (IK tay, ngắm vào khe hở của thế thủ, rút tay ngay khi chạm), **đỡ đòn** và **phản đòn**. Trọng tài tính điểm (đầu 2, thân 1), đếm knock-down (10 s là KO), tách hai võ sĩ khi ôm ghì.

Phong cách đánh (`tasks/boxing/styles.yaml`): `pressure` (áp sát), `counter` (phản đòn), `outboxer` (giữ khoảng cách). Luật trận: `tasks/boxing/exhibition.yaml`.

Học tăng cường (chiến thuật) cho máy có/không có GPU: `pip install -e '.[rl]'` rồi `python scripts/train_boxing.py --timesteps 200000`. Ghi chú chuyển sang PC NVIDIA (MuJoCo MJX): `docs/HUMANOID_BOXING.md`.

Giới hạn: chưa tự đứng dậy sau khi ngã; bị đẩy mạnh (>~20 N·s) hoặc dính đòn khi đang đứng một chân thì có thể ngã; bước chân còn chậm.

## Cảm biến của robot giúp việc

Ngoài camera RGB-D ở đầu, LiDAR 2D, IMU, encoder, cảm biến lực ở ngón kẹp và bumper, robot có thêm:

| Cảm biến | Để làm gì | Đo được |
|---|---|---|
| LiDAR 3D 16 tia trên đầu (nhìn xuống) | thấy vật thấp hơn hoặc cao hơn mặt quét LiDAR 2D | Có: vượt hộp giày 9 cm + kệ thấp, 0 va chạm. Tắt đi: chui gầm kệ, đâm, kẹt (`tasks/ablation/`) |
| 8 cảm biến ToF quanh thân | an toàn tầm gần khi docking, lùi, đi qua cửa hẹp | phát hiện khung cửa cách 6 cm |
| Camera cổ tay | căn lại vị trí vật 10 cm trước khi kẹp | sai 3–4 mm dù niềm tin ban đầu lệch 22 mm |

Vật đang kẹp trong tay được lọc khỏi dữ liệu LiDAR 3D (giống "attached object" của MoveIt), nếu không robot sẽ tưởng chính cái chai mình cầm là vật cản.

## Kiểm thử và đo lường

```bash
python -m unittest discover -s tests -v        # 62 test: đơn vị, vật lý, nhiệm vụ, nhận diện, robot boxing (test cần [vision]/[rl] tự bỏ qua nếu chưa cài)
python -m robotcode.evaluate --episodes 20     # mỗi nhiệm vụ 20 lần với vị trí ngẫu nhiên
```

Bảng kết quả có: tỉ lệ thành công, số lần robot báo thành công sai (`false+`), va chạm với nhà, va chạm với đồ vật, quãng đường, sai số định vị. Chi tiết từng lượt nằm trong `artifacts/eval_*.json`. Số liệu mới nhất: xem README.

## Mọi thứ là code / cấu hình

| Muốn thay đổi | Sửa file |
|---|---|
| Kích thước, khối lượng, motor, cảm biến, nhiễu cảm biến của robot | `robots/helper_rover.yaml` (hoặc tạo robot mới, ví dụ `robots/scout.yaml` — robot không tay) |
| Nhà, nội thất | `worlds/apartment/world.xml` |
| Tên phòng, bàn, địa điểm, từ đồng nghĩa tiếng Việt/Anh | `worlds/apartment/semantic.yaml` |
| Loại đồ vật, màu, khối lượng, chọn bộ nhận diện | `objects/catalog.yaml` |
| Nhiệm vụ, vị trí đồ vật (có ngẫu nhiên), tiêu chí thành công | `tasks/*.yaml` |

Mỗi lượt chạy được xác định hoàn toàn bởi (file nhiệm vụ, seed). File MuJoCo sinh ra nằm ở `build/scenes/`, bản đồ ở `build/maps/` (có ảnh `.pgm` kiểu Nav2).

## Bật LLM (tuỳ chọn)

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY="..."            # không commit key lên git
mjpython -m robotcode.cli --planner llm --task "Lấy giúp tôi cái cốc màu xanh trong bếp nhé"
```

LLM chỉ được viết **kế hoạch gồm các skill có sẵn**; kế hoạch được kiểm tra (skill có tồn tại không, địa điểm/đồ vật có trên bản đồ/catalog không, robot có đủ phần cứng không) trước khi robot cử động. LLM không bao giờ điều khiển motor trực tiếp. Dùng API có thể phát sinh chi phí.

## Bộ nhận diện học sâu (tuỳ chọn)

Mặc định robot tìm đồ bằng **màu** (HSV) + hình dạng từ ảnh độ sâu. Có thể đổi sang
**OWLv2** — mô hình nhận diện "từ vựng mở": gõ chữ gì (vd. "a photo of a white cup")
nó tìm khung chứa vật đó trong ảnh; phần độ sâu → toạ độ 3D → kiểm tra kích thước
vẫn dùng chung với bộ nhận diện màu. Chạy trên GPU của Mac (MPS), không cần server.

```bash
python -m pip install -e '.[vision]'        # torch + transformers; lần đầu tải ~600 MB weights
python -m robotcode.cli --headless --detector learned --task "Mang chai nước màu đỏ đến cho tao"
python -m robotcode.cli --headless --detector learned --scenario tasks/perception/white_cup_to_desk.yaml
python -m robotcode.evaluate --episodes 20 --detector learned       # so sánh với --detector color
python -m robotcode.perception_bench --episodes 5                   # 2 bộ nhận diện trên cùng ảnh, so với vị trí thật
```

Chọn bộ nhận diện: `--detector` > `perception.detector` trong `robots/*.yaml` >
`perception.detector` trong `objects/catalog.yaml` (mặc định `color`).

Điểm khác biệt: đồ **không có màu nổi bật** (trắng, đen) thì bộ màu không thấy, OWLv2
thấy được (`tasks/perception/white_cup_to_desk.yaml`). Đổi lại chậm hơn (~0,4–0,5 s/ảnh
so với ~8 ms). Kết quả đo chi tiết: README và `docs/ARCHITECTURE.md#perception`.

## Giới hạn hiện tại

- Nhận diện chỉ biết các loại đồ trong catalog (cần kích thước để tính vị trí 3D). Bộ màu cần màu đậm; OWLv2 (tuỳ chọn) nhận được màu bất kỳ nhưng chậm hơn và cần nhìn thấy gần hết vật. Chưa nhận diện người, chưa có VLM.
- Tay kiểu Cartesian + kẹp song song: gắp đồ đứng trên bàn/quầy; chưa gắp dưới sàn, chưa có tay 6 bậc.
- Bản đồ lấy từ hình học ngôi nhà (như bản vẽ mặt bằng), chưa tự lập bản đồ bằng SLAM.
- Chưa có ROS 2 / robot thật; ranh giới để chuyển sang phần cứng là `hal.py` (bảng ánh xạ ROS 2 trong `docs/ARCHITECTURE.md`).
