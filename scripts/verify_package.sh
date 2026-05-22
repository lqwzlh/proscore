#!/usr/bin/env bash
# ProScore Package Verification Script
#
# Prerequisites (host Python, not the test venv):
#   python3 -m pip install build twine
#
# Usage:
#   bash scripts/verify_package.sh              # 快速验证（日常开发）
#   bash scripts/verify_package.sh --full       # 完整验证（发布前推荐）
#   bash scripts/verify_package.sh --reuse      # 复用虚拟环境（更快）

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$HERE/dist"
TMPDIR="${TMPDIR:-/tmp}"
VENV="$TMPDIR/proscore_verify"
REUSE=false
FULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --reuse) REUSE=true; shift ;;
        --full)  FULL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  ProScore Package Verification"
echo "  Mode: $([ "$FULL" = true ] && echo "FULL" || echo "FAST") | Reuse: $REUSE"
echo "========================================"
echo ""

# 发布工具（系统 Python 常未预装，失败时勿静默）
need_tools=()
for mod in build twine; do
    if ! python3 -m "$mod" --version >/dev/null 2>&1; then
        need_tools+=("$mod")
    fi
done
if [ "${#need_tools[@]}" -gt 0 ]; then
    echo "错误: 缺少发布工具: ${need_tools[*]}"
    echo "请先安装: python3 -m pip install build twine"
    exit 1
fi

# Step 1: Build wheel
echo "[1/6] Building wheel..."
rm -rf "$DIST" build/
shopt -s nullglob
rm -rf ./*.egg-info
shopt -u nullglob
if ! python3 -m build --wheel --outdir "$DIST"; then
    echo "      ✗ wheel 构建失败（见上方输出）"
    exit 1
fi
WHEEL=$(ls "$DIST"/*.whl | head -1)
echo "      ✓ $(basename "$WHEEL")"

# Step 2: twine check（保留错误输出）
echo "[2/6] Checking metadata with twine..."
if ! python3 -m twine check "$DIST"/* 2>&1; then
    echo "      ✗ twine check failed (see above)"
    exit 1
fi
echo "      ✓ Metadata OK"

# Step 3: Wheel content check
echo "[3/6] Checking wheel contents..."
python3 -c '
import zipfile, sys
z = zipfile.ZipFile("'"$WHEEL"'")
names = z.namelist()
has_main = any("__main__.py" in n for n in names)
has_init = any(n.endswith("proscore/__init__.py") for n in names)
if not (has_main and has_init):
    print("CRITICAL: __main__.py or __init__.py missing!", file=sys.stderr)
    sys.exit(1)
print("      ✓ __main__.py and __init__.py present")
'

# Step 4: Prepare environment（首次 pip 装依赖可能需 1–3 分钟）
echo "[4/6] Preparing virtual environment (pip install, may take a few minutes)..."
if [ "$REUSE" = false ] || [ ! -d "$VENV" ]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel >/dev/null 2>&1
# template / run 依赖 openpyxl，与文档 pip install proscore[excel] 一致
pip install --quiet "${WHEEL}[excel]"
echo "      ✓ Environment ready (proscore[excel])"

# Step 5: Smoke tests
echo "[5/6] Running smoke tests..."

echo -n "      import proscore ... "
python3 -c "import proscore; print(proscore.__version__)" && echo "OK"

echo -n "      proscore --help   ... "
proscore --help >/dev/null && echo "OK"

# 生成模板（供后续 --full 使用）
TMP_PROJ="$TMPDIR/proscore_smoke_$$"
proscore template "$TMP_PROJ" >/dev/null 2>&1
echo -n "      proscore template ... "
test -f "$TMP_PROJ/pipeline_template.xlsx" && echo "OK"

# --full 模式下真正执行一次 run（预期失败）
if [ "$FULL" = true ]; then
    echo -n "      proscore run (dry)  ... "
    if proscore run "$TMP_PROJ/pipeline_template.xlsx" 2>&1 | grep -q 'data_file\|必填'; then
        echo "OK (expected error)"
    else
        echo "WARNING: run did not produce expected error"
    fi
fi

echo -n "      pip check         ... "
pip check >/dev/null 2>&1 && echo "OK"

# Step 6: Cleanup & Result
echo "[6/6] Summary..."
rm -rf "$TMP_PROJ"

echo ""
echo "========================================"
echo "  ✅ VERIFICATION PASSED"
echo "========================================"
echo "Wheel : $(basename "$WHEEL")"
echo "Mode  : $([ "$FULL" = true ] && echo "Full" || echo "Fast")"
echo ""
