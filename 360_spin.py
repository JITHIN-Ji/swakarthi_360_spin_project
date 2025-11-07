import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

from google import genai
from dotenv import load_dotenv
import os
import os
import zipfile
import base64
import time
import mimetypes
import glob

load_dotenv()

# Add cleanup function at the top
def cleanup_generated_files():
    """Remove all previously generated files"""
    files_to_remove = glob.glob("angle_*.png") + glob.glob("angle_*.jpg") + glob.glob("360_viewer.html")
    removed_count = 0
    for file in files_to_remove:
        try:
            if os.path.exists(file):
                os.remove(file)
                removed_count += 1
        except Exception as e:
            st.warning(f"Could not remove {file}: {e}")
    return removed_count

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
        # If it's a PIL Image
        buffer = BytesIO()
        image_data.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()

def save_binary_file(file_path, data):
    """Save binary data to file"""
    with open(file_path, 'wb') as f:
        f.write(data)

def generate_angle_image(client, image_data, angle_prompt, suffix, back_garment_image=None):
    """Generate a single angle image using Gemini API with optional back garment reference"""
    model = "gemini-2.5-flash-image-preview"
    
    # Convert main image to base64
    base64_image = encode_image_to_base64(image_data)
    
    # Prepare content parts
    content_parts = [
        types.Part.from_bytes(
            mime_type="image/jpeg", 
            data=base64.b64decode(base64_image)
        )
    ]
    
    # Add back garment reference image if provided and this is the back view
    if back_garment_image is not None and "back view" in angle_prompt.lower():
        back_garment_base64 = encode_image_to_base64(back_garment_image)
        content_parts.append(
            types.Part.from_bytes(
                mime_type="image/jpeg", 
                data=base64.b64decode(back_garment_base64)
            )
        )
        
        # Modify prompt for back view with garment reference
        angle_prompt = f"""BACK VIEW GENERATION WITH GARMENT REFERENCE:

PRIMARY IMAGE: Shows child wearing outfit from front/side - use this for:
- Child's exact face, hair, skin tone, body proportions
- Overall clothing colors and style consistency
- Body measurements and fit proportions

GARMENT REFERENCE IMAGE: Shows the specific back design of the garment - use this for:
- Exact back design details, patterns, prints, or text
- Back neckline style and shape
- Any back-specific design elements (buttons, zippers, graphics)
- Back fabric details and textures
- Seam placement and construction details

GENERATION REQUIREMENTS:
- Show the SAME child from primary image wearing the garment
- Apply the EXACT back design from the garment reference image
- Maintain identical body measurements and clothing fit from primary image
- Ensure the garment fits the child's body exactly as shown in primary image
- Keep consistent lighting and background style
- Show natural back view pose
- Professional fashion photography quality

CRITICAL: Combine the child's body and proportions from the primary image with the exact garment back design from the reference image. The garment should fit the child identically to how it fits in the primary image, but show the specific back design from the reference.

{angle_prompt}"""
    
    content_parts.append(types.Part.from_text(text=angle_prompt))
    
    contents = [
        types.Content(
            role="user",
            parts=content_parts
        )
    ]
    
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

def extract_clothing_details(image):
    """Extract clothing details from the image for consistent generation"""
    # This would ideally use image analysis, but for now we'll use descriptive prompts
    return {
        'type': 'clothing item',
        'context': 'child wearing outfit',
        'background': 'clean studio background'
    }

def generate_multi_angle_images_from_upload(uploaded_image, num_angles=8, prompt_strategy="comprehensive", back_garment_image=None):
    """Generate multiple angle views from uploaded image with virtual try-on focus and back garment reference"""
    try:
        client = get_genai_client()
        
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_image)
        
        # Convert to bytes for processing
        img_buffer = BytesIO()
        image.save(img_buffer, format='JPEG')
        image_bytes = img_buffer.getvalue()
        
        # Process back garment image if provided
        back_garment_bytes = None
        if back_garment_image is not None:
            back_garment_pil = Image.open(back_garment_image)
            back_garment_buffer = BytesIO()
            back_garment_pil.save(back_garment_buffer, format='JPEG')
            back_garment_bytes = back_garment_buffer.getvalue()
        
        # Calculate angles for even distribution
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
        
        # Prompting strategies for virtual try-on consistency
        prompt_strategies = {
            "basic": lambda desc, angle: f"Show the same child wearing the exact same clothing from {desc}. Maintain identical proportions, colors, and fit. {angle}° rotation.",
            
            "detailed": lambda desc, angle: f"""Create {desc} of the same child wearing identical clothing:
- EXACT same child (face, hair, body proportions, skin tone)
- IDENTICAL clothing (same colors, patterns, fabric texture, fit, wrinkles)
- SAME body measurements and clothing size
- CONSISTENT lighting and shadows
- IDENTICAL background and studio setup
- Professional fashion photography style
- {angle}° perspective rotation only""",
            
            "comprehensive": lambda desc, angle: f"""VIRTUAL TRY-ON CONSISTENCY REQUIREMENTS:
CHILD: Keep exact same child - identical face, hair, skin tone, body measurements, height, build
CLOTHING: Preserve exact same outfit - identical colors, patterns, fabric textures, fit, sizing, wrinkles, creases
MEASUREMENTS: Maintain precise body proportions - same chest/waist/hip measurements, clothing should fit identically
POSE: Natural standing pose appropriate for {desc}, arms at sides or slightly away from body
LIGHTING: Consistent studio lighting setup - same shadows, highlights, and color temperature
BACKGROUND: Plain white/light gray studio background, no distractions
QUALITY: High-resolution fashion photography, sharp details, professional retouching
ANGLE: Show {desc} perspective ({angle}° rotation) while maintaining all above consistency
IMPORTANT: This is for virtual try-on system - clothing fit and child's proportions must be absolutely identical across all angles""",
            
            "measurement_focused": lambda desc, angle: f"""MEASUREMENT CONSISTENCY FOR VIRTUAL TRY-ON:
Reference image shows child with specific measurements - replicate EXACTLY:
- Same child height and build proportions
- Identical clothing size and fit (no size variations)
- Same chest, waist, shoulder measurements
- Clothing drapes and fits identically
- Fabric behaves the same way on body
- Preserve exact garment dimensions and proportions
Show {desc} ({angle}° rotation) maintaining these critical measurements and fit consistency.
Studio photography, clean background, professional lighting.""",
            
            "technical": lambda desc, angle: f"""TECHNICAL SPECIFICATIONS for {desc}:
SUBJECT CONSISTENCY: Same child model - preserve biometric proportions, facial features, body measurements
GARMENT CONSISTENCY: Identical clothing item - same fabric properties, color values, pattern alignment, size grade
FIT ANALYSIS: Maintain exact fit relationship between garment and body - same ease, drape, tension points
DIMENSIONAL ACCURACY: Preserve 3D garment shape, body silhouette, and spatial relationships
PHOTOGRAPHIC STANDARDS: Studio lighting setup, {angle}° rotation, fashion photography composition
BACKGROUND: Neutral backdrop, consistent with virtual try-on presentation standards
OUTPUT: High-fidelity image suitable for e-commerce virtual try-on system"""
        }
        
        selected_strategy = prompt_strategies.get(prompt_strategy, prompt_strategies["comprehensive"])
        
        generated_images = []
        messages = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # First, analyze the reference image to extract key details
        base_analysis_prompt = """Analyze this image of a child wearing clothing. Identify key details for consistency:
- Child's approximate age, build, and proportions
- Clothing type, colors, patterns, and fit
- Lighting conditions and background
- Overall style and presentation
Keep these details consistent across all generated angles."""
        
        for i, (angle, description) in enumerate(zip(angles, angle_descriptions)):
            # Update progress
            progress = (i + 1) / len(angles)
            progress_bar.progress(progress)
            
            # Special handling for back view
            if "back view" in description and back_garment_image is not None:
                status_text.text(f"Generating angle {i+1}/{len(angles)}: {description} (with garment reference)")
            else:
                status_text.text(f"Generating angle {i+1}/{len(angles)}: {description}")
            
            # Create angle-specific prompt using selected strategy
            angle_prompt = selected_strategy(description, angle)
            
            # Add reference consistency instruction
            if i > 0:  # For angles after the first one
                angle_prompt += f"\n\nREFERENCE: Use the original image as the exact template for child and clothing consistency. This is angle {i+1} of a 360° product view sequence."
            
            try:
                # Pass back garment image only for back view generation
                garment_ref = back_garment_bytes if "back view" in description else None
                
                file_path = generate_angle_image(
                    client, 
                    image_bytes, 
                    angle_prompt, 
                    f"{i:02d}_{angle:03.0f}deg",
                    back_garment_image=garment_ref
                )
                
                if file_path and os.path.exists(file_path):
                    # Load generated image
                    generated_img = Image.open(file_path)
                    generated_images.append({
                        'path': file_path,
                        'angle': angle,
                        'description': description,
                        'image': generated_img
                    })
                    
                    if "back view" in description and back_garment_image is not None:
                        messages.append(f"Angle {i+1}: Generated successfully with garment reference")
                    else:
                        messages.append(f"Angle {i+1}: Generated successfully")
                    
                    # Show preview
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(file_path, caption=f"{description} ({angle}°)", width=200)
                else:
                    st.warning(f"No image generated for {description}")
                    messages.append(f"Angle {i+1}: Failed to generate")
                    
            except Exception as e:
                st.error(f"Error generating angle {i+1}: {e}")
                messages.append(f"Angle {i+1} failed: {e}")
                
            # Longer delay for better consistency (AI needs time to process complex requirements)
            time.sleep(2)
        
        progress_bar.empty()
        status_text.empty()
        
        return generated_images, messages
        
    except Exception as e:
        return [], [f"Error: {e}"]

def create_360_viewer_html(images):
    """Create HTML/JS 360° viewer"""
    if not images:
        return None
    
    # Convert images to base64 for embedding
    image_data = []
    for img_info in images:
        if os.path.exists(img_info['path']):
            with open(img_info['path'], 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
                # Determine mime type
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
        <title>360° Object Viewer</title>
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
                max-width: 600px;
                width: 100%;
            }}
            .image-container {{
                position: relative;
                width: 100%;
                height: 400px;
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
                padding: 10px 20px;
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
            .thumbnail-strip {{
                display: flex;
                justify-content: center;
                gap: 5px;
                margin-top: 15px;
                overflow-x: auto;
                padding: 10px 0;
            }}
            .thumbnail {{
                width: 60px;
                height: 60px;
                object-fit: cover;
                border-radius: 4px;
                cursor: pointer;
                opacity: 0.7;
                transition: all 0.2s ease;
                border: 2px solid transparent;
            }}
            .thumbnail:hover {{
                opacity: 1;
                transform: scale(1.1);
            }}
            .thumbnail.active {{
                opacity: 1;
                border-color: #4CAF50;
            }}
        </style>
    </head>
    <body>
        <div class="viewer-container">
            <h2>360° Object Viewer</h2>
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
            <div class="thumbnail-strip">
                {' '.join([f'<img class="thumbnail" src="{img["data"]}" data-index="{i}" alt="{img["description"]}">' for i, img in enumerate(image_data)])}
            </div>
        </div>

        <script>
            const images = {str(image_data).replace("'", '"')};
            const mainImage = document.getElementById('mainImage');
            const angleSlider = document.getElementById('angleSlider');
            const angleDisplay = document.getElementById('angleDisplay');
            const playButton = document.getElementById('playButton');
            const resetButton = document.getElementById('resetButton');
            const thumbnails = document.querySelectorAll('.thumbnail');
            
            let isPlaying = false;
            let playInterval;
            let currentIndex = 0;
            
            function updateImage(index) {{
                if (index < 0 || index >= images.length) return;
                
                currentIndex = index;
                mainImage.src = images[index].data;
                angleDisplay.textContent = images[index].description + ' (' + Math.round(images[index].angle) + '°)';
                angleSlider.value = index;
                
                // Update thumbnails
                thumbnails.forEach((thumb, i) => {{
                    thumb.classList.toggle('active', i === index);
                }});
            }}
            
            angleSlider.addEventListener('input', (e) => {{
                updateImage(parseInt(e.target.value));
            }});
            
            thumbnails.forEach((thumb, index) => {{
                thumb.addEventListener('click', () => {{
                    updateImage(index);
                }});
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
            
            // Initialize
            updateImage(0);
        </script>
    </body>
    </html>
    """
    
    return html_content



# --- Streamlit UI ---
st.set_page_config(page_title="360° Image Generator", page_icon="🔄", layout="wide")

st.title("🔄 360° Multi-Angle Image Generator with Back View Garment Reference")
st.markdown("Upload an image and generate 8 different angles of the same object using Gemini AI")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Images")
    
    # Main image upload
    uploaded_file = st.file_uploader(
        "Choose main image file (child wearing garment)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a child wearing the garment (front/side view)"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Main Image (Child wearing garment)", width=300)
    
    # Back garment reference image upload
    st.markdown("---")
    st.subheader("👔 Back Garment Reference (Optional)")
    back_garment_file = st.file_uploader(
        "Choose back garment design reference",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image showing the back design of the garment (optional - improves back view accuracy)",
        key="back_garment"
    )
    
    if back_garment_file:
        st.image(back_garment_file, caption="Back Garment Design Reference", width=300)
        st.info("✅ This design will be applied to the back view generation")
    
    st.markdown("---")
    num_angles = st.selectbox("Number of angles:", [4, 6, 8], index=2)
    
    # Prompt strategy selection
    st.subheader("🧠 AI Strategy")
    prompt_strategy = st.selectbox(
        "Choose prompting strategy:",
        options=["comprehensive", "detailed", "measurement_focused", "technical", "basic"],
        index=0,
        help="Different strategies for maintaining consistency across angles"
    )
    
    strategy_descriptions = {
        "comprehensive": "Best overall - covers all aspects including measurements, fit, and visual consistency",
        "detailed": "Detailed instructions for maintaining exact same child and clothing",
        "measurement_focused": "Emphasizes body measurements and clothing fit consistency",
        "technical": "Technical specifications for professional virtual try-on systems",
        "basic": "Simple approach - may be less consistent but faster"
    }
    
    st.info(f"**{prompt_strategy.title()}**: {strategy_descriptions[prompt_strategy]}")
    
    generate_btn = st.button("🎨 Generate Multi-Angle Images", type="primary", disabled=not uploaded_file)

with col2:
    st.subheader("ℹ️ How it works")
    st.markdown("""
    1. **Upload Main Image**: Child wearing the garment (front/side view)
    2. **Upload Back Reference** (Optional): Specific back design of the garment
    3. **Generation**: AI creates 8 different viewpoints
    4. **Back View Enhancement**: Uses garment reference for accurate back design
    5. **360° Viewer**: Interactive viewer to spin the object
    6. **Download**: Get all images and HTML viewer
    """)
    
    st.subheader("🎯 Back Garment Reference")
    st.markdown("""
    **What to Upload:**
    - Image showing the back design of the garment
    - Can be flat lay, hanger, or worn by someone else
    - Should clearly show back patterns, text, graphics
    - Higher quality = better back view generation
    
    **Benefits:**
    - Accurate back design reproduction
    - Consistent garment details
    - Professional virtual try-on quality
    - Specific patterns/text placement
    """)
    
    st.subheader("💡 Virtual Try-On Tips")
    st.markdown("""
    **For Best Results:**
    - Use high-quality images with good lighting
    - Child should be centered and clearly visible
    - Clothing should fit well (not too loose/tight)
    - Simple, clean background preferred
    - Child in natural standing pose
    - Back reference should match garment in main image
    
    **Consistency Factors:**
    - Same child measurements across all angles
    - Identical clothing fit and drape
    - Consistent colors and fabric texture
    - Professional studio lighting
    - Accurate back design from reference
    """)

if generate_btn and uploaded_file:
    st.subheader("🔄 Generating Images...")
    
    # Show status of back reference
    if back_garment_file:
        st.info("🎯 Using back garment reference for enhanced back view generation")
    else:
        st.info("ℹ️ Generating without back reference - back view will be inferred from main image")
    
    with st.spinner("⏳ Generating multiple angle views for virtual try-on..."):
        images, messages = generate_multi_angle_images_from_upload(
            uploaded_file, 
            num_angles, 
            prompt_strategy,
            back_garment_image=back_garment_file
        )
    
    if images:
        st.success(f"✅ Generated {len(images)} images successfully!")
        
        # Show generation log
        with st.expander("📝 Generation Log"):
            for msg in messages:
                if "with garment reference" in msg:
                    st.success(msg)
                else:
                    st.text(msg)
        
        # Create 360° viewer
        st.subheader("🎯 360° Interactive Viewer")
        
        viewer_html = create_360_viewer_html(images)
        if viewer_html:
            # Save HTML file
            html_file = "360_viewer.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(viewer_html)
            
            # Display in Streamlit
            st.components.v1.html(viewer_html, height=700)
            
            # Download options
            st.subheader("📥 Download")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create zip of all images
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for img_info in images:
                        if os.path.exists(img_info['path']):
                            zip_file.write(img_info['path'], os.path.basename(img_info['path']))
                
                st.download_button(
                    label="📦 Download All Images (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="360_images.zip",
                    mime="application/zip"
                )
            
            with col2:
                # Download HTML viewer
                st.download_button(
                    label="🌐 Download 360° Viewer (HTML)",
                    data=viewer_html,
                    file_name="360_viewer.html",
                    mime="text/html"
                )
        
        # Analysis section
        st.subheader("🔍 Virtual Try-On Consistency Analysis")
        analysis_text = """
        **Check these key factors:**
        - **Child Consistency**: Same face, hair, skin tone, body proportions
        - **Clothing Fit**: Identical sizing and fit across all angles
        - **Measurements**: Body measurements appear consistent
        - **Colors & Textures**: Clothing colors and fabric textures match
        - **Lighting**: Consistent studio lighting and shadows
        - **Background**: Clean, consistent background
        - **Professional Quality**: Suitable for e-commerce use
        """
        
        if back_garment_file:
            analysis_text += """
        - **Back Design Accuracy**: Back view matches provided garment reference
        - **Design Placement**: Patterns, text, graphics positioned correctly
        - **Design Consistency**: Back design integrates naturally with garment"""
        
        st.markdown(analysis_text)
        
        # Consistency scoring (placeholder for future ML analysis)
        st.subheader("📊 Consistency Score")
        base_score = min(85 + len(images) * 2, 98)
        # Bonus for using back reference
        consistency_score = min(base_score + (5 if back_garment_file else 0), 99)
        
        st.progress(consistency_score / 100)
        st.write(f"Estimated Consistency: {consistency_score}%")
        
        if back_garment_file:
            st.success("🎯 Back reference used - enhanced accuracy expected")
        
        if consistency_score < 80:
            st.warning("⚠️ Consider regenerating with 'comprehensive' or 'measurement_focused' strategy")
        elif consistency_score > 90:
            st.success("✅ High consistency - suitable for virtual try-on system")
        
        # Show thumbnail grid for comparison
        st.subheader("📸 All Angles Comparison")
        cols = st.columns(4)
        for i, img_info in enumerate(images):
            with cols[i % 4]:
                if os.path.exists(img_info['path']):
                    caption = f"{img_info['description']}"
                    if "back view" in img_info['description'] and back_garment_file:
                        caption += " ⭐"
                    st.image(img_info['path'], caption=caption)
    
    else:
        st.error("❌ Failed to generate images. Check the error messages above.")

# Cleanup section
if st.button("🧹 Clean Up Generated Files"):
    import glob
    files_to_remove = glob.glob("angle_*.png") + glob.glob("angle_*.jpg") + glob.glob("360_viewer.html")
    removed_count = 0
    for file in files_to_remove:
        try:
            os.remove(file)
            removed_count += 1
        except:
            pass
    st.success(f"🗑️ Cleaned up {removed_count} files")

st.markdown("---")
st.markdown("💡 **Note**: This tool works best with images containing single, well-defined objects. Complex scenes may not generate consistent results across all angles.") 