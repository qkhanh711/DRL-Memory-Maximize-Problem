# Hướng Dẫn Train và Phân Tích 4 Mô Hình DRL

Hệ thống hỗ trợ train và đánh giá 4 mô hình:
- **DiffPPO** (ppo_diffusion)
- **DiffQL** (ql_diffusion)  
- **PPO** (gaussian_ppo)
- **DQL** (gaussian_dql)

## Cách Sử Dụng Nhanh

### 1. Train mô hình với convergence analysis
```bash
bash run.sh train
```
Kết quả:
- Train 4 mô hình với 10 users, 1000 episodes
- Tạo convergence plots cho từng mô hình
- Lưu kết quả vào `final_plot/convergence_plots/`
- Metrics: reward, memory, latency, QoS, denoise steps, revenue

### 2. Phân tích hiệu năng với số user khác nhau
```bash
bash run.sh analyze
```
Kết quả:
- Test với 8, 10, 12 users
- So sánh hiệu năng giữa các mô hình
- Lưu plots và CSV vào `final_plot/analysis_plots/`
- Metrics: final reward, avg memory, avg latency, avg QoS, avg revenue

### 3. Phân tích QoS sweep
```bash
bash run.sh qos-sweep
```
Kết quả:
- Test với QoS targets: 25, 30, 35
- So sánh ảnh hưởng của QoS requirement
- Lưu kết quả vào `final_plot/analysis_plots/`

### 4. Chạy pipeline đầy đủ
```bash
bash run.sh full
```
Thực hiện:
1. Training với convergence analysis
2. Performance analysis
3. QoS sweep analysis

## Tùy Chỉnh Tham Số

### Thay đổi số episodes và steps
```bash
bash run.sh train --episodes 500 --max_steps 50000
```

### Thay đổi số users
```bash
bash run.sh train --num_users 15
```

### Thay đổi seed
```bash
bash run.sh train --seed 123
```

### Chọn một số mô hình cụ thể
```bash
bash run.sh train --agents ppo_diffusion gaussian_ppo
```

## Replot từ Dữ Liệu Đã Lưu

Nếu đã có dữ liệu training, có thể vẽ lại plots:

```bash
# Replot convergence
bash run.sh replot-convergence

# Replot performance analysis
bash run.sh replot-analyze
```

## Metrics Được Thu Thập

Mỗi lần training/analysis sẽ thu thập:

1. **Reward**: Tổng reward mỗi episode
2. **Memory**: Bộ nhớ sử dụng trung bình
3. **Latency**: Độ trễ trung bình (giây)
4. **QoS**: Quality of Service (BRISQUE score - càng thấp càng tốt)
5. **Denoise Steps**: Số bước denoise trung bình
6. **Revenue**: Doanh thu trung bình (từ per_user['price'])

## Kết Quả Output

Sau khi chạy, kết quả được tổ chức trong thư mục `final_plot/`:

```
final_plot/
├── convergence_plots/          # Convergence analysis
│   ├── gaussian_ppo_convergence.png
│   ├── gaussian_dql_convergence.png
│   ├── ppo_diffusion_convergence.png
│   ├── ql_diffusion_convergence.png
│   └── convergence_data.json
│
└── analysis_plots/             # Performance analysis
    ├── reward_vs_users.png
    ├── memory_vs_users.png
    ├── latency_vs_users.png
    ├── qos_vs_users.png
    ├── revenue_vs_users.png
    ├── comprehensive_analysis.csv
    └── qos_sweep_summary.csv
```

## Xem Trợ Giúp

```bash
bash run.sh help
```

## Sử Dụng main.sh

File `main.sh` có các lệnh mẫu, bạn có thể chỉnh sửa và chạy:

```bash
bash main.sh
```

Hoặc uncomment các dòng khác trong file để chạy các phân tích khác nhau.
