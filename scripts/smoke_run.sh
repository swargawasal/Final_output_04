
#!/bin/bash
echo "Running Smoke Test..."

# Test CPU Mode
export CPU_MODE=on
export GPU_MODE=off
echo "Testing CPU Mode..."
python -c "import router; print('CPU Run:', router.run_enhancement('test_input.mp4', 'test_output_cpu.mp4'))"

# Test GPU Mode (if available)
export CPU_MODE=off
export GPU_MODE=on
echo "Testing GPU Mode..."
python -c "import router; print('GPU Run:', router.run_enhancement('test_input.mp4', 'test_output_gpu.mp4'))"

echo "Done."
