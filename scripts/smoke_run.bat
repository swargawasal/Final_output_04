
@echo off
echo Running Smoke Test...

REM Test CPU Mode
set CPU_MODE=on
set GPU_MODE=off
echo Testing CPU Mode...
python -c "import router; print('CPU Run:', router.run_enhancement('test_input.mp4', 'test_output_cpu.mp4'))"

REM Test GPU Mode
set CPU_MODE=off
set GPU_MODE=on
echo Testing GPU Mode...
python -c "import router; print('GPU Run:', router.run_enhancement('test_input.mp4', 'test_output_gpu.mp4'))"

echo Done.
