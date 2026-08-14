import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_result(name, passed, message="", warn=False):
    if passed:
        print(f"{name}: [{GREEN}PASS{RESET}] {message}")
    elif warn:
        print(f"{name}: [{YELLOW}WARN{RESET}] {message}")
    else:
        print(f"{name}: [{RED}FAIL{RESET}] {message}")

def check_environment():
    all_passed = True
    
    # 1. Python version >= 3.10
    py_version = sys.version_info
    py_passed = py_version.major == 3 and py_version.minor >= 10
    print_result("Python Version (>= 3.10)", py_passed, f"{py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_passed: all_passed = False
    
    # 2. PyTorch installed
    try:
        import torch
        torch_passed = True
        print_result("PyTorch Installed", True, torch.__version__)
    except ImportError:
        torch_passed = False
        print_result("PyTorch Installed", False, "Install PyTorch: pip install torch torchvision")
        all_passed = False
        
    if not torch_passed:
        print(f"\n{RED}Overall: FAIL{RESET}")
        print("Cannot continue without PyTorch.")
        return

    # 3. CUDA available
    cuda_passed = torch.cuda.is_available()
    print_result("CUDA Available", cuda_passed, "torch.cuda.is_available()")
    if not cuda_passed: all_passed = False
    
    # 4. CUDA version
    if cuda_passed:
        print_result("CUDA Version", True, torch.version.cuda)
    
    # 5. GPU info and compute capability
    compute_cap_passed = False
    if cuda_passed:
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"
        vram_gb = props.total_memory / (1024**3)
        compute_cap_passed = (props.major >= 8) # FP8 typically requires Hopper/Blackwell, but let's check actual support
        msg = f"{props.name} (Compute Capability: {cc}, VRAM: {vram_gb:.1f}GB)"
        print_result("GPU Info", True, msg)
        if not compute_cap_passed:
            print_result("Compute Capability >= 8.9", False, f"Detected {cc}. Native FP8 requires Ada/Hopper/Blackwell (CC >= 8.9)", warn=True)

    # 6. Check torch.float8_e4m3fn
    fp8_dtype_passed = hasattr(torch, 'float8_e4m3fn')
    print_result("FP8 Dtype Support", fp8_dtype_passed, "torch.float8_e4m3fn exists" if fp8_dtype_passed else "Upgrade PyTorch to >= 2.1")
    if not fp8_dtype_passed: all_passed = False

    # 7. Test torch._scaled_mm
    scaled_mm_passed = False
    if cuda_passed and fp8_dtype_passed:
        try:
            a = torch.randn(16, 16).to(torch.float8_e4m3fn).cuda()
            b = torch.randn(16, 16).to(torch.float8_e4m3fn).cuda()
            scale_a = torch.tensor(1.0, device='cuda')
            scale_b = torch.tensor(1.0, device='cuda')
            result = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
            scaled_mm_passed = True
            print_result("torch._scaled_mm Support", True, "Successfully ran torch._scaled_mm")
        except Exception as e:
            scaled_mm_passed = False
            print_result("torch._scaled_mm Support", False, f"Failed: {str(e)}", warn=True)
            print(f"    {YELLOW}Suggestion: Your hardware may not support native FP8. Use --fp8_sim flag to run in simulation mode.{RESET}")
    else:
        print_result("torch._scaled_mm Support", False, "Skipped due to missing CUDA or FP8 dtype", warn=True)

    # Conda environment
    is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    print_result("Conda Environment", True, "Active" if is_conda else "Not Active")
    
    print("\n" + "="*40)
    if all_passed:
        if scaled_mm_passed:
            print(f"Overall: [{GREEN}PASS{RESET}] Environment is fully ready for native APA FP8 training!")
        else:
            print(f"Overall: [{YELLOW}WARN{RESET}] Environment ready for APA, but native FP8 is not supported. Use simulation mode (--fp8_sim).")
    else:
        print(f"Overall: [{RED}FAIL{RESET}] Environment has missing dependencies. See above for remediation.")

if __name__ == '__main__':
    check_environment()
