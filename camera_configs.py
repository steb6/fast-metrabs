"""
Camera intrinsic configurations for different cameras.
"""

# RealSense D435i (original values from Andrea)
REALSENSE_D435i = {
    "name": "RealSense D435i",
    "fx": 382.691528320312,
    "fy": 381.886566162109,
    "ppx": 317.998718261719,
    "ppy": 244.468139648438,
    "width": 640,
    "height": 480
}

# Webcam (calibrated with checkerboard 5x8, 11mm)
WEBCAM_CALIBRATED = {
    "name": "Webcam (Calibrated)",
    "fx": 271.038926,
    "fy": 258.575834,
    "ppx": 250.792272,
    "ppy": 233.271941,
    "width": 640,
    "height": 480,
    "calibration_error": 0.6620,  # pixels
    "distortion_coeffs": [-0.08158157, 0.01306115, 0.00598612, -0.03315828, -0.00131624]
}

# Comparison analysis
def print_comparison():
    """Print comparison between cameras"""
    print("\n" + "="*60)
    print("CAMERA INTRINSICS COMPARISON")
    print("="*60)
    
    for param in ["fx", "fy", "ppx", "ppy"]:
        rs_val = REALSENSE_D435i[param]
        wc_val = WEBCAM_CALIBRATED[param]
        diff = wc_val - rs_val
        diff_pct = (diff / rs_val) * 100
        
        print(f"\n{param:4s}:")
        print(f"  RealSense: {rs_val:10.2f}")
        print(f"  Webcam:    {wc_val:10.2f}")
        print(f"  Diff:      {diff:+10.2f} ({diff_pct:+6.1f}%)")
    
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    fx_diff = ((WEBCAM_CALIBRATED["fx"] - REALSENSE_D435i["fx"]) / REALSENSE_D435i["fx"]) * 100
    print(f"• Focal length difference: {fx_diff:.1f}%")
    print(f"  → Webcam has {'shorter' if fx_diff < 0 else 'longer'} focal length")
    print(f"  → This significantly affects depth estimation!")
    
    ppx_diff = WEBCAM_CALIBRATED["ppx"] - REALSENSE_D435i["ppx"]
    ppy_diff = WEBCAM_CALIBRATED["ppy"] - REALSENSE_D435i["ppy"]
    print(f"\n• Principal point offset: ({ppx_diff:.1f}, {ppy_diff:.1f}) pixels")
    print(f"  → This affects reprojection accuracy")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print_comparison()
