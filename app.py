from flask import Flask, request, render_template, redirect, url_for, flash
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import base64
import torch
import clip


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
EXAMPLE_FOLDER = 'static/examples'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['EXAMPLE_FOLDER'] = EXAMPLE_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(EXAMPLE_FOLDER, exist_ok=True)  

MODEL_PATH = os.path.join('models', 'best.pt')  
model = YOLO(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

EMOTION_LABELS = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised"
]

PROMPT_TEMPLATES = [
    "a photo of a person who is {}",
    "a picture showing someone who feels {}",
    "this individual appears to be {}",
    "the person is looking {}",
    "a portrait of someone who is {}"
]

def get_text_tokens(emotion_labels, prompt_templates):
    """
    Generate and tokenize text prompts for all emotions and templates.

    Args:
        emotion_labels (list): List of emotion labels.
        prompt_templates (list): List of prompt templates.

    Returns:
        tuple: A tuple containing the list of text prompts and their tokenized representations.
    """
    text_prompts = [template.format(emotion) for emotion in emotion_labels for template in prompt_templates]
    text_tokens = clip.tokenize(text_prompts).to(device)
    return text_prompts, text_tokens

text_prompts, text_tokens = get_text_tokens(EMOTION_LABELS, PROMPT_TEMPLATES)

with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_emotion(cropped_image):
    """
    Classify the emotion of the person in the cropped image using CLIP.

    Args:
        cropped_image (PIL.Image): Cropped image of the detected person.

    Returns:
        tuple: A tuple containing the top emotion label and its probability.
    """
    # Preprocess the image for CLIP
    image_input = preprocess(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)
        similarity = similarity.view(len(EMOTION_LABELS), len(PROMPT_TEMPLATES))
        aggregated_similarity = similarity.mean(dim=1)
        probs = aggregated_similarity.softmax(dim=0).cpu().numpy()

    top_prob = probs.max()
    top_emotion = EMOTION_LABELS[probs.argmax()]

    return top_emotion, top_prob

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        selected_example = request.form.get('selected_example')
        file = request.files.get('image')

        if selected_example and selected_example != '':
            example_path = os.path.join(app.config['EXAMPLE_FOLDER'], selected_example)
            if not os.path.exists(example_path):
                flash('Selected example image does not exist.')
                return redirect(request.url)
            filename = secure_filename(selected_example)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(upload_path):
                from shutil import copyfile
                copyfile(example_path, upload_path)
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
        else:
            if not selected_example:
                flash('No file part in the request.')
            else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

        try:
            results = model(upload_path)
        except Exception as e:
            flash(f'Error processing image with YOLO: {e}')
            return redirect(request.url)

        try:
            original_image = Image.open(upload_path).convert("RGB")
        except Exception as e:
            flash(f'Error opening the uploaded image: {e}')
            return redirect(request.url)

        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except IOError:
            font = ImageFont.load_default()

        for result in results:
            for detection in result.boxes:
                x1, y1, x2, y2 = detection.xyxy[0].tolist()
                cls = int(detection.cls[0])
                if cls == 0:  
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(original_image.width, x2)
                    y2 = min(original_image.height, y2)

                    draw.rectangle(((x1, y1), (x2, y2)), outline="blue", width=2)
                    
                    yolo_label = "Person"
                    
                    try:
                        bbox = draw.textbbox((x1, y1 - 10), yolo_label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except AttributeError:
                        text_width, text_height = draw.textsize(yolo_label, font=font)

                    text_background = [(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)]
                    draw.rectangle(text_background, fill="blue")

                    draw.text((x1 + 2, y1 - text_height - 2), yolo_label, fill="white", font=font)

                    cropped_image = original_image.crop((x1, y1, x2, y2))

                    top_emotion, top_prob = classify_emotion(cropped_image)

                    emotion_text = f"{top_emotion} ({top_prob*100:.1f}%)"

                    try:
                        bbox = draw.textbbox((x1, y1 + 5), emotion_text, font=font)
                        emotion_text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                    except AttributeError:
                        emotion_text_size = draw.textsize(emotion_text, font=font)

                    emotion_background = [
                        (x1, y1 + 5),
                        (x1 + emotion_text_size[0] + 4, y1 + 5 + emotion_text_size[1] + 4)
                    ]
                    draw.rectangle(emotion_background, fill="green")

                    draw.text((x1 + 2, y1 + 7), emotion_text, fill="white", font=font)

        result_image_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        try:
            annotated_image.save(result_image_path)
        except Exception as e:
            flash(f'Error saving annotated image: {e}')
            return redirect(request.url)

        try:
            with open(upload_path, "rb") as image_file:
                original_image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            with open(result_image_path, "rb") as image_file:
                result_image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            flash(f'Error encoding images: {e}')
            return redirect(request.url)

        return render_template('result.html', original_image=original_image_b64, result_image=result_image_b64, filename=filename)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',  
        debug=True,
        port=1412,
    )
