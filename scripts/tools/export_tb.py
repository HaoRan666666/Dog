"""批量导出 TensorBoard 数据到 CSV。

用法:
    python scripts/tools/export_tb.py logs/rsl_rl/RP_wd_walk_flat/
    python scripts/tools/export_tb.py logs/rsl_rl/RP_wd_walk_flat/ -t "Rewards/track_lin_vel_xy_exp"  # 只导出指定 tag
"""
import argparse
import csv
import os
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_event_file(event_path: str, tags: list[str] | None = None,
                      output_dir: str | None = None, downsample: int = 1) -> dict[str, Path]:
    """导出单个 event 文件，返回 {tag: csv_path}"""
    ea = EventAccumulator(event_path)
    ea.Reload()

    if tags is None:
        tags = ea.Tags()["scalars"]
    else:
        available = ea.Tags()["scalars"]
        tags = [t for t in tags if t in available]
        if not tags:
            print(f"  [WARN] No matching tags in {event_path}")
            return {}

    run_name = Path(event_path).parents[1].name if "run" not in str(event_path) else Path(event_path).parent.name
    out = Path(output_dir or Path(event_path).parent)
    out.mkdir(parents=True, exist_ok=True)

    saved = {}
    for tag in tags:
        events = ea.Scalars(tag)
        rows = [(e.step, e.wall_time, e.value) for e in events]
        rows = rows[::downsample]

        safe_tag = tag.replace("/", "_").replace(" ", "_")
        csv_path = out / f"{run_name}_{safe_tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "wall_time", "value"])
            writer.writerows(rows)
        saved[tag] = csv_path
        print(f"  Exported: {tag} → {csv_path} ({len(rows)} rows)")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Export TensorBoard scalars to CSV")
    parser.add_argument("log_dir", type=str, help="TensorBoard 日志根目录")
    parser.add_argument("-t", "--tag", type=str, default=None, help="只导出指定 tag（支持部分匹配）")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出目录（默认与 event 文件同目录）")
    parser.add_argument("-d", "--downsample", type=int, default=1, help="降采样因子 (1=不降采样)")
    parser.add_argument("--list", action="store_true", help="只列出所有可用 tags，不导出")
    args = parser.parse_args()

    # 找所有 event 文件
    event_files = list(Path(args.log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"[ERROR] No TensorBoard event files found in: {args.log_dir}")
        sys.exit(1)

    print(f"Found {len(event_files)} event file(s)\n")

    if args.list:
        ea = EventAccumulator(str(event_files[0]))
        ea.Reload()
        print("Available tags:")
        for tag in sorted(ea.Tags()["scalars"]):
            print(f"  {tag}")
        return

    total = 0
    for evt in sorted(event_files):
        print(f"Processing: {evt}")
        result = export_event_file(
            str(evt),
            tags=[args.tag] if args.tag else None,
            output_dir=args.output,
            downsample=args.downsample,
        )
        total += len(result)
    print(f"\nDone. Exported {total} tag(s) total.")


if __name__ == "__main__":
    main()
