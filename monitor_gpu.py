import subprocess
import time
import matplotlib.pyplot as plt
from datetime import datetime
import threading

def get_gpu_usage():
    """Get current GPU utilization percentage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0

def monitor_gpu(duration_seconds=300, interval=1):
    """Monitor GPU usage for specified duration"""
    times = []
    usage = []
    start_time = time.time()
    
    print(f"Monitoring GPU for {duration_seconds} seconds...")
    print("Time\t\tGPU Usage")
    print("-" * 30)
    
    while time.time() - start_time < duration_seconds:
        current_time = datetime.now().strftime("%H:%M:%S")
        gpu_util = get_gpu_usage()
        
        times.append(time.time() - start_time)
        usage.append(gpu_util)
        
        print(f"{current_time}\t{gpu_util:6.1f}%")
        time.sleep(interval)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(times, usage, 'b-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Utilization (%)')
    plt.title('GPU Utilization During Training')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    avg_usage = sum(usage) / len(usage)
    plt.axhline(y=avg_usage, color='r', linestyle='--', label=f'Average: {avg_usage:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gpu_utilization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSummary:")
    print(f"Average GPU Usage: {avg_usage:.1f}%")
    print(f"Peak GPU Usage: {max(usage):.1f}%")
    print(f"Minimum GPU Usage: {min(usage):.1f}%")
    
    if avg_usage < 30:
        print("\n⚠️  LOW GPU UTILIZATION DETECTED!")
        print("Possible causes:")
        print("- Network bottleneck (downloading images)")
        print("- CPU bottleneck (data loading)")
        print("- Small batch size")
        print("- Insufficient num_workers in DataLoader")
    elif avg_usage > 85:
        print("\n✅ EXCELLENT GPU UTILIZATION!")
    else:
        print("\n✅ GOOD GPU UTILIZATION")

def monitor_in_background():
    """Start monitoring in a separate thread"""
    monitor_thread = threading.Thread(target=monitor_gpu, args=(300, 2))
    monitor_thread.daemon = True
    monitor_thread.start()
    return monitor_thread

if __name__ == "__main__":
    # Monitor for 5 minutes
    monitor_gpu(duration_seconds=300, interval=2) 