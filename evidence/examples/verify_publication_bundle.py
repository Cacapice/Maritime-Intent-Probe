"""Verify a publication bundle produced by this repository."""
from __future__ import annotations
import argparse, json
from evidence.adapter import scientific_result_json_schema, evidence_graph_json_schema, verify_bundle

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("bundle")
    args=p.parse_args()
    result=verify_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1

if __name__ == "__main__": raise SystemExit(main())
