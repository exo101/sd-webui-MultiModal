import os
import gradio as gr
from modules import script_callbacks


def create_krea_angle_visualization_component(prompt_component=None):
    """
    Creates a Gradio component for 3D angle visualization using Krea AI approach.
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(current_script_dir, "camera_3d_view_krea.html")
    
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"Could not find {html_file_path}")
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # 创建iframe来嵌入3D可视化
    iframe_html = f"""
    <div id="krea-container" style="width:100%;height:500px;">
        <iframe 
            srcdoc="{html_content.replace('"', '&quot;')}" 
            width="100%" 
            height="100%" 
            frameborder="0"
            id="krea-iframe"
        ></iframe>
    </div>
    <script>
        function getKreaAngle_{len(html_content)}() {{
            const iframe = document.getElementById('krea-iframe');
            iframe.contentWindow.postMessage({{type: 'GET_CURRENT_ANGLE_KREA'}}, '*');
        }}
        
        // 监听来自iframe的角度选择结果
        window.addEventListener('message', function(event) {{
            if (event.data.type === 'ANGLE_SELECTED_KREA') {{
                // 更新Gradio组件的值
                document.getElementById('krea-azimuth-result').textContent = event.data.azimuth;
                document.getElementById('krea-elevation-result').textContent = event.data.elevation;
                document.getElementById('krea-distance-result').textContent = event.data.distance;
            }}
        }});
        
        // 监听iframe加载完成
        window.addEventListener('message', function(event) {{
            if (event.data.type === 'CAMERA_3D_READY_KREA') {{
                console.log('Krea 3D camera view ready');
            }}
        }});
    </script>
    <div style="margin-top: 10px;">
        <strong>Selected Angle:</strong>
        <span>Azimuth: </span><span id="krea-azimuth-result">front view</span>,
        <span>Elevation: </span><span id="krea-elevation-result">eye-level shot</span>,
        <span>Distance: </span><span id="krea-distance-result">medium shot</span>
    </div>
    """
    
    # 返回HTML组件和用于获取角度的按钮
    html_component = gr.HTML(iframe_html)
    
    # 定义获取角度的JS函数
    js_func = f"getKreaAngle_{len(html_content)}"
    
    # 返回组件和JS函数名
    return html_component, js_func


def create_angle_visualization_component(prompt_component=None):
    """
    Creates a Gradio component for 3D angle visualization using Krea AI approach.
    This is an alias function to maintain compatibility with flux_krea_ui.py
    """
    return create_krea_angle_visualization_component(prompt_component)


def register_krea_ui():
    """
    Registers the Krea angle selector UI components.
    """
    with gr.Blocks(analytics_enabled=False) as blocks:
        with gr.Row():
            with gr.Column():
                html_component, js_func = create_krea_angle_visualization_component()
                
                # 获取角度按钮
                get_angle_btn = gr.Button("获取当前角度", variant="primary")
                
                # 角度结果显示
                azimuth_result = gr.Textbox(label="方位角描述", interactive=False)
                elevation_result = gr.Textbox(label="高程角描述", interactive=False)
                distance_result = gr.Textbox(label="距离描述", interactive=False)
                
                # 绑定按钮事件
                get_angle_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js=js_func
                )
    
    return blocks


# 如果作为独立脚本运行，则展示UI
if __name__ == "__main__":
    register_krea_ui().launch()