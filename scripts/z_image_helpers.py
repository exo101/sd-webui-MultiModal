import os
import torch
from pathlib import Path
from typing import List, Tuple, Optional

def load_lora_safely(pipe, lora_paths: List[Tuple[str, float]], verbose: bool = True) -> bool:
    """
    安全地加载LoRA模型，处理各种错误情况
    
    Args:
        pipe: Diffusion pipeline对象
        lora_paths: LoRA路径和权重的元组列表 [(path, weight), ...]
        verbose: 是否打印详细日志
    
    Returns:
        bool: 是否成功加载至少一个LoRA
    """
    if not lora_paths:
        if verbose:
            print("[INFO] 没有LoRA模型需要加载")
        return False
    
    lora_applied = False
    
    try:
        # 清理现有的LoRA状态
        if verbose:
            print("[INFO] 清理现有LoRA状态...")
        try:
            pipe.unfuse_lora()
        except:
            pass
        try:
            pipe.unload_lora_weights()
        except:
            pass
            
    except Exception as e:
        if verbose:
            print(f"[WARNING] 清理LoRA状态时出错: {e}")
    
    # 逐个加载LoRA
    for lora_path, lora_weight in lora_paths:
        try:
            if verbose:
                print(f"[INFO] 尝试加载LoRA: {lora_path} (权重: {lora_weight})")
            
            # 验证文件存在
            if not os.path.exists(lora_path):
                if verbose:
                    print(f"[WARNING] LoRA文件不存在: {lora_path}")
                continue
                
            # 提取weight_name
            weight_name = Path(lora_path).name
            
            # 尝试不同的加载方法
            success = False
            
            # 方法1: 标准加载
            try:
                if verbose:
                    print(f"[INFO] 尝试标准加载方法...")
                pipe.load_lora_weights(lora_path, weight_name=weight_name, local_files_only=True)
                pipe.fuse_lora(lora_scale=lora_weight)
                success = True
                if verbose:
                    print(f"[SUCCESS] LoRA {weight_name} 加载成功")
            except Exception as e1:
                if verbose:
                    print(f"[INFO] 标准加载失败: {e1}")
                
                # 方法2: 不指定weight_name
                try:
                    if verbose:
                        print(f"[INFO] 尝试不指定weight_name的加载方法...")
                    pipe.load_lora_weights(lora_path, local_files_only=True)
                    pipe.fuse_lora(lora_scale=lora_weight)
                    success = True
                    if verbose:
                        print(f"[SUCCESS] LoRA {weight_name} 加载成功(方法2)")
                except Exception as e2:
                    if verbose:
                        print(f"[INFO] 方法2也失败: {e2}")
                    
                    # 方法3: 重新初始化后再加载
                    try:
                        if verbose:
                            print(f"[INFO] 尝试重新初始化后加载...")
                        # 先卸载
                        try:
                            pipe.unfuse_lora()
                            pipe.unload_lora_weights()
                        except:
                            pass
                        # 再加载
                        pipe.load_lora_weights(lora_path, weight_name=weight_name, local_files_only=True)
                        pipe.fuse_lora(lora_scale=lora_weight)
                        success = True
                        if verbose:
                            print(f"[SUCCESS] LoRA {weight_name} 加载成功(方法3)")
                    except Exception as e3:
                        if verbose:
                            print(f"[ERROR] 所有加载方法都失败: {e3}")
            
            if success:
                lora_applied = True
            else:
                if verbose:
                    print(f"[WARNING] 无法加载LoRA: {lora_path}")
                    
        except Exception as e:
            if verbose:
                print(f"[ERROR] 处理LoRA {lora_path} 时发生未知错误: {e}")
            continue
    
    if verbose:
        if lora_applied:
            print(f"[INFO] 成功加载 {sum(1 for _, _ in lora_paths if os.path.exists(_))} 个LoRA模型")
        else:
            print(f"[WARNING] 没有成功加载任何LoRA模型")
    
    return lora_applied

def get_available_lora_models(models_path: str) -> List[str]:
    """
    获取可用的LoRA模型列表
    
    Args:
        models_path: 模型根路径
        
    Returns:
        List[str]: LoRA模型名称列表
    """
    lora_path = Path(models_path) / "Lora"
    if not lora_path.exists():
        return []
    
    lora_files = []
    # 查找所有支持的LoRA文件
    for ext in ['.safetensors', '.ckpt', '.pt']:
        lora_files.extend([f.stem for f in lora_path.glob(f"*{ext}")])
    
    # 去重并排序
    unique_loras = list(set(lora_files))
    return sorted(unique_loras)

def validate_lora_path(lora_name: str, models_path: str) -> Optional[str]:
    """
    验证LoRA模型路径是否存在
    
    Args:
        lora_name: LoRA模型名称
        models_path: 模型根路径
        
    Returns:
        Optional[str]: 完整路径，如果不存在则返回None
    """
    lora_path = Path(models_path) / "Lora"
    
    # 检查各种可能的扩展名
    for ext in ['.safetensors', '.ckpt', '.pt']:
        full_path = lora_path / f"{lora_name}{ext}"
        if full_path.exists():
            return str(full_path)
    
    return None