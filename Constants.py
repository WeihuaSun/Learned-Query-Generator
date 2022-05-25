from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = "postgres://postgres:1@192.168.31.102:20432/imdbnew"
DATA_ROOT = Path("./data")
OUTPUT_ROOT = Path("./output")
MODEL_ROOT = Path("./model")
MULTI_TABLE = DATA_ROOT / "multi-table"
SINGLE_TABLE = DATA_ROOT / "singe-table"
BNFBASE = DATA_ROOT/"bnf-lang"
PKL_PROTO = 4
__name__ = "abc"

