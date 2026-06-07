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
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--train-scope",
        choices=["full", "head"],
        default="full",
        help="full trains all layers; head freezes all layers before the final YOLO head.",
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=None,
        help="Freeze the first N Ultralytics model layers. Overrides --train-scope head when set.",
    )
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate passed to Ultralytics.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate factor passed to Ultralytics.")
    parser.add_argument("--optimizer", default=None, help="Optimizer passed to Ultralytics, for example auto, SGD, Adam, AdamW.")
    parser.add_argument("--cos-lr", action="store_true", help="Enable Ultralytics cosine learning rate schedule.")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs passed to Ultralytics.")
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
        "patience": args.patience,
        "train_scope": args.train_scope,
        "freeze_layers": args.freeze_layers,
        "effective_freeze": None,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "optimizer": args.optimizer,
        "cos_lr": args.cos_lr,
        "warmup_epochs": args.warmup_epochs,
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

    if args.freeze_layers is not None and args.freeze_layers < 0:
        payload["errors"].append("--freeze-layers must be 0 or greater.")
        die()

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
            effective_freeze = args.freeze_layers
            if effective_freeze is None and args.train_scope == "head":
                layers = getattr(getattr(model, "model", None), "model", None)
                if layers is None:
                    payload["errors"].append("Unable to infer model layers for --train-scope head. Use --freeze-layers N.")
                    die()
                effective_freeze = max(len(layers) - 1, 0)
            payload["effective_freeze"] = effective_freeze

            train_kwargs = {
                "data": args.data,
                "project": resolved_project,
                "name": args.name,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "patience": args.patience,
                "exist_ok": True,
                "verbose": not args.json,
            }
            if effective_freeze is not None:
                train_kwargs["freeze"] = effective_freeze
            if args.lr0 is not None:
                train_kwargs["lr0"] = args.lr0
            if args.lrf is not None:
                train_kwargs["lrf"] = args.lrf
            if args.optimizer is not None:
                train_kwargs["optimizer"] = args.optimizer
            if args.cos_lr:
                train_kwargs["cos_lr"] = True
            if args.warmup_epochs is not None:
                train_kwargs["warmup_epochs"] = args.warmup_epochs

            model.train(
                **train_kwargs
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
