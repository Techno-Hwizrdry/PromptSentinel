import argparse
import sys
from prompt_sentinel.detector import PromptSentinel

def main():
    parser = argparse.ArgumentParser(description="PromptSentinel: ML-Powered Injection Detection")
    parser.add_argument("--scan", type=str, help="The prompt text to analyze")
    args = parser.parse_args()

    if not args.scan:
        parser.print_help()
        sys.exit()
    
    print("Loading model onto GPU...")
    sentinel = PromptSentinel("./fine_tuned_sentinel")
    print("Done!  PromptSentinel is ready.  Scanning prompt text...\n")

    result = sentinel.scan(args.scan)
    print(f"Result: {result}")
        
if __name__ == "__main__":
    main()