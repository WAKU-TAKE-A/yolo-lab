import os
import sys
import json
import argparse
import shutil
from contextlib import contextmanager

try:
    from ultralytics import YOLO
except ImportError:
    print(json.dumps({"errors": ["ultralytics package is not installed."]}))
    sys.exit(1)

@contextmanager
def suppress_process_stdio(enabled):
    if not enabled:
        yield
        return

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    with open(os.devnull, "w") as devnull:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    resolved_project = os.path.abspath(args.project)
    run_dir = os.path.abspath(os.path.join(resolved_project, args.name))

    payload = {
        "model_path": args.model,
        "data_path": args.data,
        "project": resolved_project,
        "name": args.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "run_dir": None,
        "best_model": None,
        "last_model": None,
        "results_csv": None,
        "args_yaml": None,
        "warnings": [],
        "errors": []
    }
    
    def die():
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for e in payload["errors"]:
                print(e, file=sys.stderr)
        sys.exit(1 if payload["errors"] else 0)

    if not os.path.exists(args.model):
        payload["errors"].append(f"Model path does not exist: {args.model}")
        die()
        
    if not os.path.exists(args.data):
        payload["errors"].append(f"Data yaml path does not exist: {args.data}")
        die()
        
    if os.path.exists(run_dir):
        if not args.force:
            payload["errors"].append(f"Output run directory already exists: {run_dir}. Use --force to overwrite.")
            die()
        else:
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                payload["errors"].append(f"Failed to remove existing run directory: {e}")
                die()
                
    try:
        with suppress_process_stdio(args.json):
            model = YOLO(args.model)
            model.train(
                data=args.data,
                project=resolved_project,
                name=args.name,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                exist_ok=True,
                verbose=not args.json
            )
    except Exception as e:
        payload["errors"].append(f"Training failed: {str(e)}")
            
    if payload["errors"]:
        die()
        
    payload["run_dir"] = run_dir
    
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        payload["best_model"] = best_pt
        
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    if os.path.exists(last_pt):
        payload["last_model"] = last_pt
        
    res_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(res_csv):
        payload["results_csv"] = res_csv
        
    args_yaml = os.path.join(run_dir, "args.yaml")
    if os.path.exists(args_yaml):
        payload["args_yaml"] = args_yaml
        
    if args.json:
        # Provide one clean json object
        print(json.dumps(payload, indent=2))
    else:
        print(f"Training complete. Run dir: {run_dir}")
        if payload["best_model"]:
            print(f"Best model: {payload['best_model']}")

if __name__ == "__main__":
    main()
