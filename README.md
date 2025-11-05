# Human Pose Estimator

Real-time 3D human pose estimation using TensorRT and RGB cameras.

![Demo](demo.gif)

## 🚀 Quick Start

### 1. Setup Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate human_pose_estimator
```

After creating the environment, install PyTorch with CUDA support:

```bash
# For CUDA 12.6 (adjust cu126 to match your CUDA version)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Note:** Find CUDA version and get the appropriate PyTorch install command.

### 2. Download Model Files

Download the required ONNX models and skeleton conversion files from:

**[IIT Dataverse - Human Pose Estimation Models](https://dataverse.iit.it/dataset.xhtml?persistentId=doi%3A10.48557%2FKFMAIC&version=DRAFT)**

Extract the downloaded files into the `assets/` folder.

### 3. Convert Models to TensorRT

Run the setup scripts to convert ONNX models to TensorRT engines:

```bash
cd setup
python create_engine_bbone.py
cd ..
```
Do it for all files in the setup folder (e.g. `bbone`, `heads`, `image_transformation`,  `yolo`).
This will generate optimized `.engine` files in the `assets/` directory.

## 📹 Usage

### Option A: Standalone Mode (Webcam)

Run the inference script directly to connect to your webcam:

```bash
python inference.py
```

**Customize behavior:** Edit `inference.py` to:
- Use a video file instead of webcam (set `VIDEO_PATH` variable)
- Adjust camera settings
- Modify visualization options

### Option B: YARP Module Mode

For integration with robotic systems or distributed applications:

1. **Start YARP server** (in terminal 1):
   ```bash
   yarpserver
   ```

2. **Launch the pose estimation module** (in terminal 2):
   ```bash
   python yarp_module.py
   ```

3. **[Optional] Test with video file** (in terminal 3):
   ```bash
   python yarp_source.py
   ```
   This streams frames from `input.mp4` to the YARP module.

4. **Connect ports manually** (if needed):
   ```bash
   # Connect video source to pose estimator
   yarp connect /test_yarp/image:o /hpe/image:i
   
   # View annotated output
   yarpview --name /viewer
   yarp connect /hpe/image:o /viewer
   
   # Read pose data
   yarp read ... /hpe/pose:o
   ```

## 📊 Output

The system provides:
- **3D skeleton visualization** in real-time (matplotlib window)
- **Annotated video stream** with bounding boxes
- **Pose data** including:
  - 3D joint positions (x, y, z)
  - Human distance from camera
  - Human position in camera coordinates
  - Skeleton edges for rendering

## 🛠️ Configuration

Edit `configs.py` to customize:
- Camera intrinsics (fx, fy, ppx, ppy)
- Detection thresholds
- Model paths
- Skeleton type

## 📋 Requirements

- CUDA-capable GPU (CUDA 11.x or 12.x)
- YARP (optional, for distributed mode)

## 🔧 Troubleshooting

**TensorRT engine errors:**
Re-run the setup scripts to rebuild engines for your specific GPU and TensorRT version.

**No camera detected:**
- Check camera permissions
- Verify camera index in `inference.py`
- Or use video file mode by setting `VIDEO_PATH`
