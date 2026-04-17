import json

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Rectangle

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
JSON_PATH = "precision_search_results.json"
OUTPUT_PATH = "winner_matrix_speedup.png"

ORDER = [
    "trt_int8",
    "trt_fp16",
    "trt_bf16",
    "trt_fp8",
    "compile_fp32",
    "eager_fp32",
    "no valid",
]

PRETTY = {
    "trt_int8": "TRT INT8",
    "trt_fp16": "TRT FP16",
    "trt_bf16": "TRT BF16",
    "trt_fp8": "TRT FP8",
    "compile_fp32": "torch.compile",
    "eager_fp32": "Eager FP32",
    "no valid": "Evaluation failed",
}

NEUTRAL_VARIANTS = {"eager_fp32", "no valid"}
NEUTRAL_COLOR = "#e5e7eb"
EDGE_COLOR = "#666666"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def model_sort_key(name: str):
    prefix = name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else 9999


def shorten(name: str) -> str:
    mapping = {
        "DenseNet121TransitionLayer": "DenseNet121Trans",
        "DenseNet121DenseBlock": "DenseNet121Block",
        "ConvolutionalVisionTransformer": "CVT",
        "VisionTransformer": "ViT",
        "GoogleNetInceptionV1": "GoogLeNet",
        "GoogleNetInceptionModule": "GoogLeNetMod",
        "MinGPTCausalAttention": "MinGPTAttn",
        "Mamba2ReturnFinalState": "Mamba2Final",
        "Mamba2ReturnY": "Mamba2Y",
        "ShallowWideMLP": "WideMLP",
        "DeepNarrowMLP": "DeepMLP",
        "EfficientNetMBConv": "EffNetMBConv",
        "SwinTransformerV2": "SwinV2",
        "VisionAttention": "VisionAttn",
        "NetVladWithGhostClusters": "NetVladGhost",
        "NetVladNoGhostClusters": "NetVlad",
        "ReLUSelfAttention": "ReLUSelfAttn",
        "GRUBidirectionalHidden": "GRUBiHidden",
        "GRUBidirectional": "GRUBi",
        "LSTMBidirectional": "LSTMBi",
    }
    idx, rest = name.split("_", 1)
    return f"{idx} {mapping.get(rest, rest)}"


def nice_speedup_text(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.2f}x"


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

rows = []
for r in results:
    winner = r.get("best_valid_overall_variant")
    if winner is None:
        winner = "no valid"

    eager_latency = r.get("eager_latency_ms")
    compile_latency = r.get("compile_fp32_latency_ms")
    lowp_speedup = r.get("speedup_over_eager_from_lowp", 1.0)

    compile_speedup = None
    if eager_latency is not None and compile_latency is not None and compile_latency > 0:
        compile_speedup = eager_latency / compile_latency

    # Speedup to display/color by winner type
    if winner == "compile_fp32":
        display_speedup = compile_speedup
    elif winner in {"trt_fp16", "trt_bf16", "trt_fp8", "trt_int8"}:
        display_speedup = lowp_speedup
    elif winner == "eager_fp32":
        display_speedup = 1.0
    else:
        display_speedup = None

    rows.append({
        "model_name": r["model_name"],
        "display_name": shorten(r["model_name"]),
        "winner": winner,
        "display_speedup": display_speedup,
    })

rows = sorted(rows, key=lambda x: model_sort_key(x["model_name"]))

# Group rows by winning variant
grouped = {k: [] for k in ORDER}
for row in rows:
    grouped[row["winner"]].append(row)

max_rows = max(len(grouped[k]) for k in ORDER)

# Collect speedups for colored variants
colored_speedups = [
    row["display_speedup"]
    for row in rows
    if row["winner"] not in NEUTRAL_VARIANTS
       and row["winner"] != "no valid"
       and row["display_speedup"] is not None
       and row["display_speedup"] > 1.0
]

vmin = 1.0
vmax = max(colored_speedups) if colored_speedups else 2.0
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.get_cmap("YlOrRd")  # yellow -> orange -> red

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, max(10, 0.58 * max_rows)))

for col, variant in enumerate(ORDER):
    models = grouped[variant]

    for row_idx in range(max_rows):
        y = max_rows - 1 - row_idx

        # Base cell
        ax.add_patch(Rectangle(
            (col, y), 1, 1,
            facecolor="white",
            edgecolor="#b0b0b0",
            linewidth=1.0
        ))

        if row_idx < len(models):
            item = models[row_idx]

            # Color logic
            if variant in NEUTRAL_VARIANTS or variant == "no valid":
                facecolor = NEUTRAL_COLOR
            else:
                sp = item["display_speedup"]
                facecolor = cmap(norm(sp)) if sp is not None else NEUTRAL_COLOR

            ax.add_patch(Rectangle(
                (col, y), 1, 1,
                facecolor=facecolor,
                edgecolor=EDGE_COLOR,
                linewidth=1.3
            ))

            speedup_text = ""
            if item["display_speedup"] is not None:
                speedup_text = f"\n{nice_speedup_text(item['display_speedup'])}"

            ax.text(
                col + 0.5,
                y + 0.54,
                f"{item['display_name']}{speedup_text}",
                ha="center",
                va="center",
                fontsize=14,
                linespacing=1.0,
            )

# Axes
ax.set_xlim(0, len(ORDER))
ax.set_ylim(0, max_rows)
ax.set_xticks([i + 0.5 for i in range(len(ORDER))])
ax.set_xticklabels([PRETTY[k] for k in ORDER], fontsize=15, rotation=20, ha="right")
ax.set_yticks([])

ax.set_title("Best performing precision / backend per model \n(color = speedup over eager)", fontsize=18, pad=18)
ax.set_xlabel("Winning precision / backend", fontsize=15)

# Strong grid
for x in range(len(ORDER) + 1):
    ax.axvline(x, color="#6b7280", linewidth=1.5)
for y in range(max_rows + 1):
    ax.axhline(y, color="#6b7280", linewidth=1.1)

for spine in ax.spines.values():
    spine.set_visible(False)

# Colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Speedup over eager", fontsize=15)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=260, bbox_inches="tight")
plt.show()
