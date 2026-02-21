import gradio as gr
import os
import base64
from pathlib import Path

def create_qwen_image_edit_angle_visualization_component(text_to_image_prompt_component):
    """为qwen图像编辑创建独立的3D视角可视化选择器"""
    
    # 读取专用的3D相机HTML文件
    html_file_path = Path(__file__).parent / "camera_3d_view_qwen_edit.html"
    
    # 如果专用HTML文件不存在，则抛出错误
    if not html_file_path.exists():
        raise FileNotFoundError(f"3D相机HTML文件不存在: {html_file_path}")
    
    # 读取HTML内容并转换为base64编码
    with open(html_file_path, 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()
    
    # 将HTML内容转换为data URI
    encoded_content = base64.b64encode(html_content.encode()).decode()
    data_uri = f"data:text/html;base64,{encoded_content}"
    
    # 创建Gradio HTML组件展示3D相机界面
    with gr.Row():
        gr.HTML(f'<iframe id="camera-3d-view-qwen-edit" src="{data_uri}" width="100%" height="500px" style="border: none;"></iframe>')
    
    # 创建应用角度到提示词的按钮
    apply_btn = gr.Button("应用角度到提示词", variant="primary")
    
    # 角度选择结果文本框（用于调试和内部处理）
    angle_result = gr.Textbox(visible=False, interactive=False)
    
    # 使用JavaScript实现按钮点击事件
    apply_btn.click(
        fn=None,
        inputs=[text_to_image_prompt_component],  # 输入提示词组件
        outputs=[text_to_image_prompt_component], # 输出到提示词组件
        _js="""
        async (prompt) => {
            console.log('Qwen Edit Apply button clicked, prompt:', prompt); // 添加错误检查
            
            // 获取iframe元素
            const iframe = document.getElementById('camera-3d-view-qwen-edit');
            
            console.log('Qwen Edit Iframe element:', iframe); // 添加错误检查
            
            // 确保iframe已经加载完成
            if (!iframe || !iframe.contentWindow) {
                console.error('无法访问qwen edit iframe内容，可能尚未加载完成');
                alert('错误：无法访问3D视角界面，请刷新页面后重试！');
                return [prompt]; // 返回原始提示词，不添加任何角度信息
            }
            
            // 等待iframe加载完成
            await new Promise(resolve => {
                if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
                    resolve();
                } else {
                    iframe.onload = resolve;
                    // 设置超时，避免无限等待
                    setTimeout(resolve, 1000);
                }
            });
            
            console.log('Sending message to qwen edit iframe...'); // 添加错误检查
            
            // 向iframe发送请求角度信息的消息
            iframe.contentWindow.postMessage({
                type: 'GET_CURRENT_ANGLE_QWEN_EDIT'
            }, '*');
            
            console.log('Message sent to qwen edit iframe, waiting for response...'); // 添加错误检查
            
            // 等待响应
            return new Promise((resolve) => {
                const startTime = Date.now();
                const timeoutId = setTimeout(() => {
                    const elapsed = Date.now() - startTime;
                    // 超时处理
                    window.removeEventListener('message', handleMessage);
                    console.log(`等待qwen edit角度数据超时 (${elapsed}ms)`);
                    alert(`错误：等待qwen edit 3D视角界面响应超时 (${elapsed}ms)，请检查3D界面是否正常工作！`);
                    resolve([prompt]); // 返回原始提示词，不添加任何角度信息
                }, 3000); // 增加超时时间到3秒
                
                const handleMessage = (event) => {
                    const elapsed = Date.now() - startTime;
                    console.log(`Received message from qwen edit iframe after ${elapsed}ms:`, event.data); // 添加错误检查
                    
                    // 确保消息来源是我们的qwen edit iframe
                    if (event.data.type === 'ANGLE_SELECTED_QWEN_EDIT') {
                        console.log('接收到qwen edit角度数据:', event.data); // 调试信息
                        clearTimeout(timeoutId);
                        window.removeEventListener('message', handleMessage);
                        
                        // 构建角度提示词，包含所有非空的角度信息
                        const parts = ['<sks>'];
                        if (event.data.azimuth && event.data.azimuth !== "") parts.push(event.data.azimuth);
                        if (event.data.elevation && event.data.elevation !== "") parts.push(event.data.elevation);
                        if (event.data.distance && event.data.distance !== "") parts.push(event.data.distance);
                        
                        console.log('Qwen Edit Parts after filtering:', parts); // 添加错误检查
                        console.log('Qwen Edit Event data received:', event.data); // 添加错误检查
                        
                        // 检查是否至少有一个角度信息（不仅仅是<sks>）
                        if (parts.length <= 1) {
                            console.log('没有接收到有效的qwen edit角度数据');
                            console.log('Azimuth:', event.data.azimuth, 'Elevation:', event.data.elevation, 'Distance:', event.data.distance); // 添加错误检查
                            console.log('Data types - Azimuth:', typeof event.data.azimuth, 'Elevation:', typeof event.data.elevation, 'Distance:', typeof event.data.distance); // 检查数据类型
                            
                            // 直接报错而不是使用默认值
                            alert('错误：未能从qwen edit 3D视角界面获取到有效的角度数据，请检查3D界面是否正常工作！');
                            resolve([prompt]); // 返回原始提示词，不添加任何角度信息
                            return;
                        }
                        
                        const anglePrompt = parts.join(' ');
                        
                        // 更新主提示词输入框
                        let newPrompt = prompt;
                        if (prompt.includes('<sks>')) {
                            // 替换现有的<sks>部分
                            const regex = /<sks>[^<]*(?:<(?!sks>|\/sks>)[^<]*)*/g;
                            newPrompt = prompt.replace(regex, anglePrompt);
                        } else {
                            // 追加到现有提示词
                            newPrompt = prompt + (prompt ? ' ' : '') + anglePrompt;
                        }
                        
                        console.log('Qwen Edit 新提示词:', newPrompt); // 调试信息
                        resolve([newPrompt]);
                    } else {
                        console.log('Received non-ANGLE_SELECTED_QWEN_EDIT message:', event.data); // 添加错误检查
                    }
                };
                
                window.addEventListener('message', handleMessage);
            });
        }
        """
    )
    
    return apply_btn, angle_result

# 检查是否可用的标志
QWEN_IMAGE_EDIT_ANGLE_SELECTOR_AVAILABLE = True