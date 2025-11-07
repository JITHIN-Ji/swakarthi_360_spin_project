import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import base64
import mimetypes
import concurrent.futures
from threading import Lock

load_dotenv()

@st.cache_resource
def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

def encode_image_to_base64(image_data):
    """Convert image data to base64"""
    if isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode()
    else:
        buffer = BytesIO()
        image_data.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()

def save_binary_file(file_path, data):
    """Save binary data to file"""
    with open(file_path, 'wb') as f:
        f.write(data)

def generate_angle_image(client, image_data, angle_prompt, suffix, back_garment_image=None):
    """Generate a single angle image using Gemini API"""
    model = "gemini-2.5-flash-image-preview"
    
    base64_image = encode_image_to_base64(image_data)
    
    content_parts = [
        types.Part.from_bytes(
            mime_type="image/jpeg", 
            data=base64.b64decode(base64_image)
        )
    ]
    
    if back_garment_image is not None and "back view" in angle_prompt.lower():
        back_garment_base64 = encode_image_to_base64(back_garment_image)
        content_parts.append(
            types.Part.from_bytes(
                mime_type="image/jpeg", 
                data=base64.b64decode(back_garment_base64)
            )
        )
    
    content_parts.append(types.Part.from_text(text=angle_prompt))
    
    contents = [types.Content(role="user", parts=content_parts)]
    
    config = types.GenerateContentConfig(
        response_modalities=["image", "text"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE")
        ],
        response_mime_type="text/plain",
    )
    
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if part.inline_data:
                        file_name = f"angle_{suffix}"
                        inline_data = part.inline_data
                        ext = mimetypes.guess_extension(inline_data.mime_type) or '.png'
                        file_path = f"{file_name}{ext}"
                        save_binary_file(file_path, inline_data.data)
                        return file_path
        return None
    except Exception as e:
        st.error(f"Error generating angle image: {e}")
        return None

def generate_single_angle(args):
    """Wrapper function for concurrent execution"""
    client, image_bytes, angle, description, index, back_garment_bytes = args
    
    angle_prompt = f"""Generate {description} of the same child wearing identical clothing.
Maintain exact same child appearance, clothing fit, colors, and proportions.
Show {angle}° rotation perspective."""
    
    garment_ref = back_garment_bytes if "back view" in description else None
    
    file_path = generate_angle_image(
        client, 
        image_bytes, 
        angle_prompt, 
        f"{index:02d}_{angle:03.0f}deg",
        back_garment_image=garment_ref
    )
    
    if file_path and os.path.exists(file_path):
        generated_img = Image.open(file_path)
        return {
            'path': file_path,
            'angle': angle,
            'description': description,
            'image': generated_img,
            'index': index
        }
    return None

def generate_multi_angle_concurrent(uploaded_image, back_garment_image=None, num_angles=8):
    """Generate multiple angles concurrently"""
    try:
        client = get_genai_client()
        
        image = Image.open(uploaded_image)
        img_buffer = BytesIO()
        image.save(img_buffer, format='JPEG')
        image_bytes = img_buffer.getvalue()
        
        back_garment_bytes = None
        if back_garment_image is not None:
            back_garment_pil = Image.open(back_garment_image)
            back_garment_buffer = BytesIO()
            back_garment_pil.save(back_garment_buffer, format='JPEG')
            back_garment_bytes = back_garment_buffer.getvalue()
        
        angles = [i * (360 / num_angles) for i in range(num_angles)]
        angle_descriptions = [
            "front view",
            "front-right diagonal view", 
            "right side view",
            "back-right diagonal view",
            "back view",
            "back-left diagonal view",
            "left side view",
            "front-left diagonal view"
        ]
        
        # Prepare arguments for concurrent execution
        tasks = [
            (client, image_bytes, angle, desc, i, back_garment_bytes)
            for i, (angle, desc) in enumerate(zip(angles, angle_descriptions))
        ]
        
        generated_images = []
        
        # Execute concurrently with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(generate_single_angle, task): i for i, task in enumerate(tasks)}
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                progress = completed / len(futures)
                progress_bar.progress(progress)
                status_text.text(f"Generating: {completed}/{len(futures)} angles completed")
                
                result = future.result()
                if result:
                    generated_images.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort by index to maintain order
        generated_images.sort(key=lambda x: x['index'])
        
        return generated_images
        
    except Exception as e:
        st.error(f"Error: {e}")
        return []

def create_360_viewer_html(images):
    """Create HTML/JS 360° viewer"""
    if not images:
        return None
    
    image_data = []
    for img_info in images:
        if os.path.exists(img_info['path']):
            with open(img_info['path'], 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
                mime_type = "image/png"
                if img_info['path'].lower().endswith(('.jpg', '.jpeg')):
                    mime_type = "image/jpeg"
                
                image_data.append({
                    'data': f"data:{mime_type};base64,{img_b64}",
                    'angle': img_info['angle'],
                    'description': img_info['description']
                })
    
    if not image_data:
        return None
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>360° Viewer</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
                background: #f0f0f0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .viewer-container {{
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                padding: 20px;
                max-width: 800px;
                width: 100%;
            }}
            .image-container {{
                position: relative;
                width: 100%;
                height: 500px;
                display: flex;
                justify-content: center;
                align-items: center;
                background: #fafafa;
                border-radius: 8px;
                overflow: hidden;
            }}
            .main-image {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                transition: opacity 0.2s ease;
            }}
            .controls {{
                margin-top: 20px;
                text-align: center;
            }}
            .angle-slider {{
                width: 100%;
                margin: 10px 0;
                height: 8px;
                border-radius: 4px;
                background: #ddd;
                outline: none;
            }}
            .angle-slider::-webkit-slider-thumb {{
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #4CAF50;
                cursor: pointer;
            }}
            .angle-info {{
                margin: 10px 0;
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }}
            .play-button {{
                background: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
                transition: background 0.2s;
            }}
            .play-button:hover {{
                background: #45a049;
            }}
            .play-button.stop {{
                background: #f44336;
            }}
            .play-button.stop:hover {{
                background: #da190b;
            }}
        </style>
    </head>
    <body>
        <div class="viewer-container">
            <div class="image-container">
                <img id="mainImage" class="main-image" src="{image_data[0]['data']}" alt="360° View">
            </div>
            <div class="controls">
                <div class="angle-info">
                    <span id="angleDisplay">{image_data[0]['description']} ({int(image_data[0]['angle'])}°)</span>
                </div>
                <input type="range" id="angleSlider" class="angle-slider" 
                       min="0" max="{len(image_data)-1}" value="0" step="1">
                <div>
                    <button id="playButton" class="play-button">▶ Auto Rotate</button>
                    <button id="resetButton" class="play-button">🔄 Reset</button>
                </div>
            </div>
        </div>

        <script>
            const images = {str(image_data).replace("'", '"')};
            const mainImage = document.getElementById('mainImage');
            const angleSlider = document.getElementById('angleSlider');
            const angleDisplay = document.getElementById('angleDisplay');
            const playButton = document.getElementById('playButton');
            const resetButton = document.getElementById('resetButton');
            
            let isPlaying = false;
            let playInterval;
            let currentIndex = 0;
            
            function updateImage(index) {{
                if (index < 0 || index >= images.length) return;
                
                currentIndex = index;
                mainImage.src = images[index].data;
                angleDisplay.textContent = images[index].description + ' (' + Math.round(images[index].angle) + '°)';
                angleSlider.value = index;
            }}
            
            angleSlider.addEventListener('input', (e) => {{
                updateImage(parseInt(e.target.value));
            }});
            
            playButton.addEventListener('click', () => {{
                if (isPlaying) {{
                    clearInterval(playInterval);
                    playButton.textContent = '▶ Auto Rotate';
                    playButton.classList.remove('stop');
                    isPlaying = false;
                }} else {{
                    playInterval = setInterval(() => {{
                        currentIndex = (currentIndex + 1) % images.length;
                        updateImage(currentIndex);
                    }}, 500);
                    playButton.textContent = '⏸ Stop';
                    playButton.classList.add('stop');
                    isPlaying = true;
                }}
            }});
            
            resetButton.addEventListener('click', () => {{
                if (isPlaying) {{
                    clearInterval(playInterval);
                    playButton.textContent = '▶ Auto Rotate';
                    playButton.classList.remove('stop');
                    isPlaying = false;
                }}
                updateImage(0);
            }});
            
            updateImage(0);
        </script>
    </body>
    </html>
    """
    
    return html_content

# --- Streamlit UI ---
st.set_page_config(page_title="360° Viewer Generator", page_icon="🔄", layout="centered")

st.title("Swakriti 360 Viewer")

# Upload sections
uploaded_file = st.file_uploader(
    "Upload Person Image",
    type=['png', 'jpg', 'jpeg']
)

back_garment_file = st.file_uploader(
    "Upload Back Garment Reference (Optional)",
    type=['png', 'jpg', 'jpeg']
)

# Generate button
if st.button("🎨 Generate 360° Viewer", type="primary", disabled=not uploaded_file):
    with st.spinner("⏳ Generating 360° view..."):
        images = generate_multi_angle_concurrent(
            uploaded_file, 
            back_garment_image=back_garment_file
        )
    
    if images:
        st.success(f"✅ Generated {len(images)} angles!")
        
        # Add heading for the viewer
        st.markdown("---")
        st.subheader("🎯 360° Object Viewer")
        
        # Create and display 360° viewer
        viewer_html = create_360_viewer_html(images)
        if viewer_html:
            st.components.v1.html(viewer_html, height=700)
    else:
        st.error("❌ Failed to generate images.")
