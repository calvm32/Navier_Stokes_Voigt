import argparse
import subprocess
import sys
import os
import shutil
import re

def build_parser():
    parser = argparse.ArgumentParser(
        description="Automated benchmarking wrapper for mysave and myrun"
    )

    # ---------------------------
    # mysave identical arguments
    # ---------------------------
    parser.add_argument("--problem", required=True, 
                        choices=["h2-fem", "h3-fem", "nse2-fem", "nsv2-fem", "comp2-fem", 
                                 "h2-spec", "nse2-spec", "nsv2-spec", "comp-spec"], 
                        help="Problem type")
    
    parser.add_argument("--mms", action="store_true", 
                        help="Use method of manufactured solutions")
    
    parser.add_argument("--elements", choices=["sv", "th"], 
                        help="Element type")
    
    parser.add_argument("--mesh", 
                        help="Optional name of mesh [with corresp. file extension]")
    
    parser.add_argument("--set", action="append", default=[], metavar="FILE.KEY=VALUE", 
                        help="Override yaml values")

    # --------------------------------
    # Benchmark-specific arguments
    # --------------------------------
    parser.add_argument("--cores", type=int, default=14, 
                        help="Number of cores to use (default: 14)")
    
    parser.add_argument("--dt", type=float, default=0.0001, 
                        help="Time step size (default: 0.0001)")
    
    parser.add_argument("--alpha", type=float, default=0.07, 
                        help="Alpha parameter (default: 0.07)")
    
    parser.add_argument("--test-steps", type=int, default=1000, 
                        help="Number of steps for the short benchmark (default: 1000)")
    
    parser.add_argument("--final-time", type=float, default=100.0, 
                        help="Target final time T for production (default: 100.0)")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Calculate targets based on inputs
    test_T = args.test_steps * args.dt
    total_steps = int(args.final_time / args.dt)
    dir_name = f"benchmark_{args.cores}cores"

    print("==== ACCESS Benchmarking Test ====")
    print("Testing the following...")
    print(f"Problem: {args.problem}")
    print(f"Elements: {args.elements}")
    print(f"Mesh: {args.mesh}\n")
    print("Testing with...")
    print(f"Cores: {args.cores}")
    print(f"Test steps: {args.test_steps} (dt={args.dt}, test_T={test_T})")
    print(f"Production Target: T={args.final_time} (Total steps: {total_steps:,})")

    # Clean up old directory
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    print("\nBuilding run folder...")
    
    # Construct the mysave command
    mysave_cmd = ["mysave", dir_name, "--problem", args.problem]
    if args.mms:
        mysave_cmd.append("--mms")
    if args.elements:
        mysave_cmd.extend(["--elements", args.elements])
    if args.mesh:
        mysave_cmd.extend(["--mesh", args.mesh])
    
    # Add user-specified --set overrides
    for s in args.set:
        mysave_cmd.extend(["--set", s])
        
    # Append the benchmarking overrides (these will execute last)
    mysave_cmd.extend([
        "--set", f"user_settings.alpha={args.alpha}",
        "--set", f"user_settings.dt={args.dt}",
        "--set", f"user_settings.T={test_T}"
    ])

    # Execute mysave
    try:
        subprocess.run(mysave_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error: mysave failed. Check your arguments.")
        sys.exit(1)

    print("\nExecuting solver...")
    
    # Construct the myrun command
    myrun_cmd = ["myrun", "--np", str(args.cores), dir_name]
    
    extracted_minutes = None
    
    # Execute myrun, capture output live to terminal and to log file
    with open("run_output.log", "w") as logfile:
        
        # Popen allows us to stream the output continuously instead of waiting for the end
        process = subprocess.Popen(
            myrun_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")  # Print to terminal
            logfile.write(line)  # Write to file
            
            # Use regex to dynamically hunt for the completion time string
            match = re.search(r'Completed after ([\d.]+) minutes', line)
            if match:
                extracted_minutes = float(match.group(1))
                
        process.wait()

    if extracted_minutes is None:
        print("\nError: Could not extract solver time. Did the run fail?")
        sys.exit(1)

    # Compute Metrics
    elapsed_sec = extracted_minutes * 60.0
    sec_per_step = elapsed_sec / args.test_steps
    hours_per_full_run = (sec_per_step * total_steps) / 3600.0
    core_hours_per_run = hours_per_full_run * args.cores

    # Print Table Data
    print("\n==========================================")
    print("DATA GENERATED")
    print("==========================================")
    print(f"Time for {args.test_steps} steps:       {elapsed_sec:.2f} seconds")
    print(f"Time per step:                {sec_per_step:.5f} seconds\n")
    print(f"Target Final Time (T):        {args.final_time}")
    print(f"Total projected steps:        {total_steps:,}")
    print(f"Hours/run:                    {hours_per_full_run:.2f} h")
    print(f"Core-hours per run:           {core_hours_per_run:.2f}")
    print("==========================================")

if __name__ == "__main__":
    main()