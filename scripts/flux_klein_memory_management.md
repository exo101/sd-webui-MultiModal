# FLUX.2-klein 生成器 - 内存管理优化

## 问题描述
在使用 FLUX.2-klein 模型进行图像生成、编辑或扩展后，模型资源未被正确释放，导致显存持续占用。

## 解决方案
添加了 `cleanup_pipeline()` 函数，并在所有生成函数的 `finally` 块中调用该函数，确保无论成功还是失败都能释放资源。

## 修改内容

### 1. 新增清理函数
```python
def cleanup_pipeline(pipe):
    """清理模型管道，释放显存和系统资源"""
```

**清理步骤**：
1. 卸载 LoRA 权重（如果已加载）
2. 将所有组件移到 CPU（transformer、text_encoder、text_encoder_2、vae）
3. 删除管道引用
4. 清空 CUDA 缓存
5. 重置 CUDA 种子
6. 执行 Python 垃圾回收

### 2. 修改的函数

#### `generate_flux_klein_image()`
- 初始化 `pipe = None`
- 在 `finally` 块中调用 `cleanup_pipeline(pipe)`

#### `multi_img_flux_klein()`
- 初始化 `pipe = None`
- 在 `finally` 块中调用 `cleanup_pipeline(pipe)`

#### `inpaint_flux_klein()`
- 初始化 `pipe = None`
- 在 `finally` 块中调用 `cleanup_pipeline(pipe)`

#### `extend_flux_klein()`
- 初始化 `pipe = None`
- 在 `finally` 块中调用 `cleanup_pipeline(pipe)`

## 资源释放流程

```
生成完成 → finally 块触发
        ↓
    cleanup_pipeline() 执行
        ↓
1. 卸载 LoRA 权重
2. 模型组件移到 CPU
   - transformer.cpu()
   - text_encoder.cpu()
   - text_encoder_2.cpu()
   - vae.cpu()
3. del pipe (删除引用)
4. torch.cuda.empty_cache()
5. gc.collect() (垃圾回收)
        ↓
    显存释放完成
```

## 效果
- ✅ 生成完成后立即释放显存
- ✅ 支持连续多次生成而不爆显存
- ✅ 异常情况下也能正确清理资源
- ✅ 详细的清理日志便于调试

## 日志输出示例
```
Starting pipeline cleanup...
Unloaded LoRA weights
Moved transformer to CPU
Moved text_encoder to CPU
Freed CUDA cache
Garbage collection completed
Pipeline cleanup completed successfully
```
