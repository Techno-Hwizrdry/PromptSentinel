import time
import torch
from prompt_sentinel.detector import PromptSentinel

def main() -> None:
    print("Initializing benchmark on GPU...")
    sentinel = PromptSentinel("./fine_tuned_sentinel")
    
    test_prompt = "Explain the history of the internet."
    iterations = 100
    
    # Warm up the GPU (CUDA initialization)
    _ = sentinel.scan(test_prompt)
    
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        _ = sentinel.scan(test_prompt)
        
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) / iterations) * 1000
    print(f"\n--- Benchmark Results ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {1000/avg_latency:.1f} prompts/sec")

if __name__ == "__main__":
    main()