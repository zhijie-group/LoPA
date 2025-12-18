"""
统计功能模块：用于收集和分析生成过程中的性能数据

功能：
1. 以数据集为单位，统计峰值 TPS (tokens per second)
2. 以数据集为单位，统计每个样例在每一步的 TPF (tokens per forward/step)
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

eval_logger = logging.getLogger(__name__)


class GenerationStatsCollector:
    """收集生成过程中的统计信息"""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        self.sample_stats: List[Dict[str, Any]] = []
        self.dataset_stats: Dict[str, Any] = {}
        
    def record_sample(
        self,
        sample_idx: int,
        tokens: int,
        steps: int,
        generation_time: float,
        tpf_per_step: Optional[List[float]] = None,
        dataset_name: Optional[str] = None,
    ):
        """记录单个样例的统计信息
        
        Args:
            sample_idx: 样例索引
            tokens: 生成的 token 数量
            steps: 生成步数
            generation_time: 生成耗时（秒）
            tpf_per_step: 每一步的 TPF 列表（tokens per forward）
            dataset_name: 数据集名称
        """
        tps = tokens / generation_time if generation_time > 0 else 0.0
        avg_tpf = tokens / steps if steps > 0 else 0.0
        
        sample_stat = {
            "sample_idx": sample_idx,
            "tokens": tokens,
            "steps": steps,
            "generation_time": generation_time,
            "tps": tps,
            "avg_tpf": avg_tpf,
            "tpf_per_step": tpf_per_step or [],
            "dataset_name": dataset_name,
        }
        self.sample_stats.append(sample_stat)
        
    def compute_dataset_stats(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """计算数据集级别的统计信息
        
        Args:
            dataset_name: 数据集名称，如果为 None 则统计所有样例
            
        Returns:
            包含峰值 TPS 等统计信息的字典
        """
        if dataset_name:
            relevant_samples = [s for s in self.sample_stats if s.get("dataset_name") == dataset_name]
        else:
            relevant_samples = self.sample_stats
            
        if not relevant_samples:
            return {}
            
        tps_list = [s["tps"] for s in relevant_samples]
        peak_tps = max(tps_list) if tps_list else 0.0
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        min_tps = min(tps_list) if tps_list else 0.0
        
        # 计算最快前10个样例的TPS均值
        top10_tps_mean = 0.0
        if tps_list:
            sorted_tps = sorted(tps_list, reverse=True)
            top10_count = min(10, len(sorted_tps))
            top10_tps_mean = sum(sorted_tps[:top10_count]) / top10_count if top10_count > 0 else 0.0
        
        # 找到最高速的样例信息
        peak_sample = None
        if relevant_samples:
            peak_sample_idx = max(range(len(relevant_samples)), key=lambda i: relevant_samples[i]["tps"])
            peak_sample = {
                "sample_idx": relevant_samples[peak_sample_idx]["sample_idx"],
                "tps": relevant_samples[peak_sample_idx]["tps"],
                "tokens": relevant_samples[peak_sample_idx]["tokens"],
                "steps": relevant_samples[peak_sample_idx]["steps"],
                "generation_time": relevant_samples[peak_sample_idx]["generation_time"],
            }
        
        total_tokens = sum(s["tokens"] for s in relevant_samples)
        total_steps = sum(s["steps"] for s in relevant_samples)
        total_time = sum(s["generation_time"] for s in relevant_samples)
        
        overall_tps = total_tokens / total_time if total_time > 0 else 0.0
        overall_tpf = total_tokens / total_steps if total_steps > 0 else 0.0
        
        stats = {
            "dataset_name": dataset_name or "all",
            "num_samples": len(relevant_samples),
            "peak_tps": peak_tps,
            "peak_sample": peak_sample,  # 最高速样例的详细信息
            "top10_tps_mean": top10_tps_mean,  # 最快前10个样例的TPS均值
            "avg_tps": avg_tps,
            "min_tps": min_tps,
            "overall_tps": overall_tps,
            "overall_tpf": overall_tpf,
            "total_tokens": total_tokens,
            "total_steps": total_steps,
            "total_time": total_time,
        }
        
        if dataset_name:
            self.dataset_stats[dataset_name] = stats
        else:
            self.dataset_stats["all"] = stats
            
        return stats
    
    def build_tpf_table(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """构建每个样例在每一步的 TPF 表格
        
        Args:
            dataset_name: 数据集名称，如果为 None 则包含所有样例
            
        Returns:
            TPF 表格，每行包含样例索引、步数和对应的 TPF
        """
        if dataset_name:
            relevant_samples = [s for s in self.sample_stats if s.get("dataset_name") == dataset_name]
        else:
            relevant_samples = self.sample_stats
            
        tpf_table = []
        for sample in relevant_samples:
            sample_idx = sample["sample_idx"]
            tpf_per_step = sample.get("tpf_per_step", [])
            dataset_name_actual = sample.get("dataset_name", "unknown")
            
            for step_idx, tpf in enumerate(tpf_per_step):
                tpf_table.append({
                    "sample_idx": sample_idx,
                    "step": step_idx + 1,  # 从1开始计数
                    "tpf": tpf,
                    "dataset_name": dataset_name_actual,
                })
                
        return tpf_table
    
    def save_stats(self, output_path: Optional[str] = None):
        """保存统计信息到文件
        
        Args:
            output_path: 输出文件路径，如果为 None 则使用 save_dir
        """
        if output_path is None:
            if self.save_dir is None:
                eval_logger.warning("No save_dir specified, skipping stats save")
                return
            os.makedirs(self.save_dir, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
            output_path = os.path.join(self.save_dir, f"generation_stats_{timestamp}.json")
        
        # 计算所有数据集的统计信息
        datasets = set(s.get("dataset_name") for s in self.sample_stats if s.get("dataset_name"))
        for dataset_name in datasets:
            self.compute_dataset_stats(dataset_name)
        self.compute_dataset_stats()  # 计算总体统计
        
        # 构建 TPF 表格
        tpf_table = self.build_tpf_table()
        
        output_data = {
            "sample_stats": self.sample_stats,
            "dataset_stats": self.dataset_stats,
            "tpf_table": tpf_table,
            "timestamp": time.time(),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        eval_logger.info(f"Statistics saved to {output_path}")
        
        # 同时保存 CSV 格式的 TPF 表格（便于分析）
        if tpf_table:
            csv_path = output_path.replace(".json", "_tpf_table.csv")
            import csv
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_idx", "step", "tpf", "dataset_name"])
                writer.writeheader()
                writer.writerows(tpf_table)
            eval_logger.info(f"TPF table saved to {csv_path}")


def load_stats_from_file(stats_file: str) -> Dict[str, Any]:
    """从文件加载统计信息
    
    Args:
        stats_file: 统计文件路径
        
    Returns:
        包含统计信息的字典
    """
    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_peak_tps(stats_data: Dict[str, Any], dataset_name: Optional[str] = None) -> Dict[str, float]:
    """分析峰值 TPS
    
    Args:
        stats_data: 从 load_stats_from_file 加载的数据
        dataset_name: 数据集名称，如果为 None 则分析所有数据集
        
    Returns:
        包含峰值 TPS 信息的字典
    """
    dataset_stats = stats_data.get("dataset_stats", {})
    
    if dataset_name:
        if dataset_name in dataset_stats:
            return {
                "dataset": dataset_name,
                "peak_tps": dataset_stats[dataset_name]["peak_tps"],
                "avg_tps": dataset_stats[dataset_name]["avg_tps"],
                "overall_tps": dataset_stats[dataset_name]["overall_tps"],
            }
        else:
            eval_logger.warning(f"Dataset {dataset_name} not found in stats")
            return {}
    else:
        # 返回所有数据集的峰值 TPS
        result = {}
        for name, stats in dataset_stats.items():
            result[name] = {
                "peak_tps": stats["peak_tps"],
                "avg_tps": stats["avg_tps"],
                "overall_tps": stats["overall_tps"],
            }
        return result


def get_tpf_table(stats_data: Dict[str, Any], dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取 TPF 表格
    
    Args:
        stats_data: 从 load_stats_from_file 加载的数据
        dataset_name: 数据集名称，如果为 None 则返回所有样例
        
    Returns:
        TPF 表格列表
    """
    tpf_table = stats_data.get("tpf_table", [])
    
    if dataset_name:
        return [row for row in tpf_table if row.get("dataset_name") == dataset_name]
    else:
        return tpf_table


if __name__ == "__main__":
    # 示例用法
    collector = GenerationStatsCollector(save_dir="./stats_output")
    
    # 模拟一些数据
    collector.record_sample(
        sample_idx=0,
        tokens=100,
        steps=10,
        generation_time=2.5,
        tpf_per_step=[10.0, 12.0, 8.0, 11.0, 9.0, 10.0, 12.0, 8.0, 11.0, 9.0],
        dataset_name="test_dataset",
    )
    
    collector.record_sample(
        sample_idx=1,
        tokens=150,
        steps=15,
        generation_time=3.0,
        tpf_per_step=[10.0] * 15,
        dataset_name="test_dataset",
    )
    
    # 计算统计信息
    stats = collector.compute_dataset_stats("test_dataset")
    print("Dataset stats:", json.dumps(stats, indent=2))
    
    # 保存统计信息
    collector.save_stats()

