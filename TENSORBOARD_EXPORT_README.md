# TensorBoard Export Tools

This directory contains scripts to export TensorBoard training data as images and create interactive HTML visualizations.

## 📁 Files

- **`export_tensorboard_images.py`**: Basic export script that exports all metrics as PNG images
- **`export_tensorboard_advanced.py`**: Advanced exporter with filtering and customization options
- **`view_tensorboard_exports.py`**: HTML viewer generator for browsing exported images

## 🚀 Quick Start

### 1. Export All Metrics (Basic)

```bash
python export_tensorboard_images.py
```

This will:
- Export all metrics from `logs/rsl_rl/` to `tensorboard_exports/`
- Create individual plots for each metric
- Generate a comparison plot with key metrics

### 2. Export with Filtering (Advanced)

```bash
# Export specific task
python export_tensorboard_advanced.py --task evobot_v1_velocity

# Export specific run
python export_tensorboard_advanced.py --run 2026-01-15_17-04-57

# Export specific metrics only
python export_tensorboard_advanced.py --metrics "Train/mean_reward,Loss/value_function"

# Export with custom smoothing
python export_tensorboard_advanced.py --smooth 0.8
```

### 3. Generate HTML Viewer

```bash
python view_tensorboard_exports.py
```

Then open in your browser:
```
file:///home/hongquan/Documents/GitHub/IsaacLabUTE/tensorboard_exports/index.html
```

## 📊 Usage Examples

### Export Specific Task Training Results

```bash
# Export evobot_v1_velocity runs
python export_tensorboard_advanced.py \
    --task evobot_v1_velocity \
    --output tensorboard_exports/evobot_velocity

# Generate HTML viewer
python view_tensorboard_exports.py \
    --export-dir tensorboard_exports/evobot_velocity/evobot_v1_velocity
```

### Compare Multiple Tasks

```bash
# Export all tasks
python export_tensorboard_images.py --logdir logs/rsl_rl --output all_tasks

# Export specific tasks separately
for task in evobot_v1_velocity anymal_c_flat anymal_c_navigation; do
    python export_tensorboard_advanced.py \
        --task $task \
        --output tensorboard_exports/$task
done
```

### Export Key Metrics Only

```bash
# Training performance metrics
python export_tensorboard_advanced.py \
    --metrics "Train/mean_reward,Train/mean_episode_length,Loss/value_function,Loss/surrogate"

# Reward components
python export_tensorboard_advanced.py \
    --metrics "Episode_Reward/lin_vel_tracking,Episode_Reward/ang_vel_tracking,Episode_Reward/rpy_alignment"
```

### Export Latest Run

```bash
# Find latest run
latest_run=$(ls -t logs/rsl_rl/evobot_v1_velocity/ | head -1)

# Export it
python export_tensorboard_advanced.py \
    --task evobot_v1_velocity \
    --run $latest_run \
    --output tensorboard_exports/latest_run
```

## 🎨 Features

### Basic Export Script

- **Simple one-command export**: No configuration needed
- **Smoothed plots**: Exponential moving average applied
- **Comparison plots**: Automatic selection of key metrics
- **Multi-run support**: Overlays all runs on same plot

### Advanced Export Script

- **Task filtering**: Export specific task only
- **Run filtering**: Export specific training run
- **Metric filtering**: Export only specified metrics
- **Custom smoothing**: Adjustable smoothing factor
- **Grid comparison**: 3x3 grid of key metrics
- **Summary report**: Text file with export statistics

### HTML Viewer

- **Dark theme**: Easy on the eyes
- **Categorized view**: Metrics organized by type
- **Click to zoom**: Full-screen image viewer
- **Smooth navigation**: Quick jump to sections
- **Responsive design**: Works on mobile

## 📋 Command Line Options

### `export_tensorboard_images.py`

```
--logdir PATH       TensorBoard logs directory (default: logs/rsl_rl)
--output PATH       Output directory (default: tensorboard_exports)
--smooth FLOAT      Smoothing factor 0-1 (default: 0.6)
```

### `export_tensorboard_advanced.py`

```
--logdir PATH       TensorBoard logs directory (default: logs/rsl_rl)
--output PATH       Output directory (default: tensorboard_exports)
--task NAME         Filter by task name (e.g., evobot_v1_velocity)
--run TIMESTAMP     Filter by run timestamp (e.g., 2026-01-15_17-04-57)
--metrics LIST      Comma-separated metric names to export
--smooth FLOAT      Smoothing factor 0-1 (default: 0.6)
--no-grid           Skip grid comparison plot
```

### `view_tensorboard_exports.py`

```
--export-dir PATH   Directory containing exported images (default: tensorboard_exports)
```

## 📸 Output Structure

```
tensorboard_exports/
├── index.html                              # HTML viewer
├── comparison_plot.png                     # Multi-metric comparison
├── metrics_grid.png                        # Grid layout of key metrics
├── export_summary.txt                      # Export statistics
├── Episode_Reward_alive.png               # Individual metric plots
├── Episode_Reward_terminating.png
├── Loss_value_function.png
├── Train_mean_reward.png
└── ...
```

## 🔧 Dependencies

Required Python packages (already installed with Isaac Lab):
- `tensorboard`
- `matplotlib`
- `numpy`

## 💡 Tips

1. **Faster exports**: Use `--task` to filter specific task instead of exporting all
2. **Publication plots**: Increase DPI in script (default: 200)
3. **Custom colors**: Modify color schemes in script source
4. **Batch processing**: Use shell loops to export multiple tasks
5. **Storage**: PNG files are ~100-300KB each, plan accordingly for large exports

## 🐛 Troubleshooting

### "No event files found"
- Check that `--logdir` points to correct location
- Verify TensorBoard logs exist: `ls -la logs/rsl_rl/`

### "Module not found: tensorboard"
- Install: `pip install tensorboard`

### Images not loading in HTML viewer
- Use absolute paths
- Check file permissions
- Open HTML file in browser directly (File > Open)

### Plots look too noisy
- Increase `--smooth` parameter (try 0.8 or 0.9)

### Out of memory
- Export one task at a time using `--task`
- Export fewer metrics using `--metrics`

## 📖 Examples Gallery

### Training Progress Comparison
```bash
python export_tensorboard_advanced.py \
    --task evobot_v1_velocity \
    --metrics "Train/mean_reward,Train/mean_episode_length" \
    --output training_progress
```

### Reward Component Analysis
```bash
python export_tensorboard_advanced.py \
    --task evobot_v1_velocity \
    --metrics "Episode_Reward/lin_vel_tracking,Episode_Reward/ang_vel_tracking,Episode_Reward/rpy_alignment,Episode_Reward/action_rate" \
    --output reward_analysis
```

### Loss Curves
```bash
python export_tensorboard_advanced.py \
    --metrics "Loss/value_function,Loss/surrogate,Loss/entropy" \
    --output loss_curves
```

## 🤝 Contributing

Feel free to extend these scripts with:
- PDF export support
- More plot types (histograms, distributions)
- LaTeX export for papers
- Animation/video generation
- Real-time monitoring

## 📝 License

These scripts follow the Isaac Lab license (BSD-3-Clause).
