import gradio as gr
import os
from modules import scripts
import urllib.parse

def create_qwen_angle_visualization_component(prompt_component):
    """
    为qwen_image_edit创建独立的多角度可视化选择器，使用3D相机视角界面
    """
    with gr.Group():
        gr.Markdown("### 3D相机控制调整视角")
        
        # 获取3D可视化界面的路径
        html_path = os.path.join(os.path.dirname(__file__), "camera_3d_view_qwen.html")
        
        # 使用相对路径，让Gradio处理静态文件服务
        # 创建一个隐藏的iframe来加载3D界面
        with gr.Row():
            # 使用Gradio的文件服务，而不是file://协议
            gr.HTML(f'''
            <div style="position:relative; width:100%; height:400px;">
                <iframe id="camera-iframe-qwen" name="camera-iframe-qwen" 
                        src="/file={urllib.parse.quote(html_path)}" 
                        width="100%" height="100%" 
                        style="border: 1px solid #444; border-radius: 8px;"></iframe>
            </div>
            ''')
        
        # 应用按钮
        with gr.Row():
            apply_btn = gr.Button("应用角度到提示词", variant="primary")
            
            # JavaScript代码用于从3D界面接收消息并应用到提示词
            apply_js = """
            async (prompt) => {
                console.log('Qwen Apply button clicked, prompt:', prompt); // 添加错误检查
                
                // 获取iframe元素
                const iframe = document.getElementById('camera-iframe-qwen');
                
                console.log('Qwen Iframe element:', iframe); // 添加错误检查
                
                // 确保iframe已经加载完成
                if (!iframe || !iframe.contentWindow) {
                    console.error('无法访问qwen iframe内容，可能尚未加载完成');
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
                
                console.log('Sending message to qwen iframe...'); // 添加错误检查
                
                // 向iframe发送请求角度信息的消息
                iframe.contentWindow.postMessage({
                    type: 'GET_CURRENT_ANGLE_QWEN'
                }, '*');
                
                console.log('Message sent to qwen iframe, waiting for response...'); // 添加错误检查
                
                // 等待响应
                return new Promise((resolve) => {
                    const startTime = Date.now();
                    const timeoutId = setTimeout(() => {
                        const elapsed = Date.now() - startTime;
                        // 超时处理
                        window.removeEventListener('message', handleMessage);
                        console.log(`等待qwen角度数据超时 (${elapsed}ms)`);
                        alert(`错误：等待qwen 3D视角界面响应超时 (${elapsed}ms)，请检查3D界面是否正常工作！`);
                        resolve([prompt]); // 返回原始提示词，不添加任何角度信息
                    }, 3000); // 增加超时时间到3秒
                    
                    const handleMessage = (event) => {
                        const elapsed = Date.now() - startTime;
                        console.log(`Received message from qwen iframe after ${elapsed}ms:`, event.data); // 添加错误检查
                        
                        // 确保消息来源是我们的qwen iframe
                        if (event.data.type === 'ANGLE_SELECTED_QWEN') {
                            console.log('接收到qwen角度数据:', event.data); // 调试信息
                            clearTimeout(timeoutId);
                            window.removeEventListener('message', handleMessage);
                            
                            // 构建角度提示词，包含所有非空的角度信息
                            const parts = ['<sks>'];
                            if (event.data.azimuth && event.data.azimuth !== "") parts.push(event.data.azimuth);
                            if (event.data.elevation && event.data.elevation !== "") parts.push(event.data.elevation);
                            if (event.data.distance && event.data.distance !== "") parts.push(event.data.distance);
                            
                            console.log('Qwen Parts after filtering:', parts); // 添加错误检查
                            console.log('Qwen Event data received:', event.data); // 添加错误检查
                            
                            // 检查是否至少有一个角度信息（不仅仅是<sks>）
                            if (parts.length <= 1) {
                                console.log('没有接收到有效的qwen角度数据');
                                console.log('Azimuth:', event.data.azimuth, 'Elevation:', event.data.elevation, 'Distance:', event.data.distance); // 添加错误检查
                                console.log('Data types - Azimuth:', typeof event.data.azimuth, 'Elevation:', typeof event.data.elevation, 'Distance:', typeof event.data.distance); // 检查数据类型
                                
                                // 直接报错而不是使用默认值
                                alert('错误：未能从qwen 3D视角界面获取到有效的角度数据，请检查3D界面是否正常工作！');
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
                            
                            console.log('Qwen 新提示词:', newPrompt); // 调试信息
                            resolve([newPrompt]);
                        } else {
                            console.log('Received non-ANGLE_SELECTED_QWEN message:', event.data); // 添加错误检查
                        }
                    };
                    
                    window.addEventListener('message', handleMessage);
                });
            }
            """
            
            apply_btn.click(
                fn=None,
                inputs=[prompt_component],
                outputs=[prompt_component],
                _js=apply_js
            )

    return prompt_component