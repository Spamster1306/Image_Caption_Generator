from flask import Flask, render_template, request, redirect
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

# Load the pre-trained ViT model and set up the feature extractor and tokenizer
vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the pre-trained Blip model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
blip_model.to(device)

max_length_vit = 16
num_beams_vit = 4
gen_kwargs_vit = {"max_length": max_length_vit, "num_beams": num_beams_vit}

max_length_blip = 32
num_beams_blip = 5
gen_kwargs_blip = {"max_length": max_length_blip, "num_beams": num_beams_blip}

# Define a route to render the HTML form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        caption_method = request.form.get("caption_method")  # Get the selected caption method
        reference_caption = request.form.get("reference_caption")  # Get the reference caption
        action = request.form.get("action")  # Get the action (Generate Caption or Evaluate)
        if file and action:
            image_path, caption, bleu_graph_path, meteor_graph_path, table_path = "", "", "", "", ""
            vit_caption, blip_caption = "", ""
            if action == "Generate Caption":
                if caption_method == "vit":
                    image_path, caption = predict_caption_vit(file)
                    return render_template("result_vit.html", image_path=image_path, caption=caption)
                elif caption_method == "blip":
                    image_path, caption = predict_caption_blip(file)
                    return render_template("result_blip.html", image_path=image_path, caption=caption)
                else:
                    caption = "Invalid caption method selected."
                    return render_template("error.html", message=caption)
            elif action == "Evaluate":
                # Add logic to evaluate models based on the uploaded image and reference caption
                image_path, reference_caption, vit_caption, blip_caption,  bleu_graph_path, meteor_graph_path, table_path = evaluate_models(file, reference_caption)
                return render_template("evaluation_result.html", image_path = image_path, reference_caption= reference_caption, vit_caption = vit_caption
                                       , blip_caption = blip_caption, bleu_graph_path = bleu_graph_path, meteor_graph_path = meteor_graph_path, table_path = table_path)
    return render_template("index.html")

# Function to predict a caption for an image using ViT
def predict_caption_vit(file):
    i_image = Image.open(file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = vit_feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = vit_model.generate(pixel_values, **gen_kwargs_vit)
    preds = vit_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    # Save the uploaded image to a file
    image_filename = file.filename
    image_path = os.path.join("static", image_filename)
    i_image.save(image_path)

    # Return the relative image path and the caption
    return image_filename, preds[0]

# Function to predict a caption for an image using Blip
def predict_caption_blip(file):
    i_image = Image.open(file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    inputs = blip_processor(i_image, return_tensors="pt")
    inputs.to(device)

    output_ids = blip_model.generate(**inputs, **gen_kwargs_blip)
    preds = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    
    # Save the uploaded image to a file
    image_filename = file.filename
    image_path = os.path.join("static", image_filename)
    i_image.save(image_path)

    # Return the relative image path and the caption
    return image_filename, preds

# Function to evaluate models based on the uploaded image and reference caption
def evaluate_models(file, reference_caption):
    # Open the input image
    i_image = Image.open(file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    # Process the input image using ViT model
    vit_pixel_values = vit_feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    vit_pixel_values = vit_pixel_values.to(device)
    vit_output_ids = vit_model.generate(vit_pixel_values, **gen_kwargs_vit)
    vit_preds = vit_tokenizer.batch_decode(vit_output_ids, skip_special_tokens=True)
    vit_caption = vit_preds[0]

    # Process the input image using BLIP model
    blip_inputs = blip_processor(i_image, return_tensors="pt").to(device)
    blip_output_ids = blip_model.generate(**blip_inputs, **gen_kwargs_blip)
    blip_preds = blip_processor.decode(blip_output_ids[0], skip_special_tokens=True)
    blip_caption = blip_preds

    reference_caption_tokens = word_tokenize(reference_caption)
    vit_caption_tokens = word_tokenize(vit_caption)
    blip_caption_tokens = word_tokenize(blip_caption)

    # Calculate BLEU score for ViT model
    vit_bleu_score = sentence_bleu([reference_caption.split()], vit_caption.split())
    vit_meteor_score = single_meteor_score(reference_caption_tokens, vit_caption_tokens)


    # Calculate BLEU score for BLIP model
    blip_bleu_score = sentence_bleu([reference_caption.split()], blip_caption.split())
    blip_meteor_score = single_meteor_score(reference_caption_tokens, blip_caption_tokens)

    # Create a bar graph
    models = ['ViT', 'BLIP']
    bleu_scores = [vit_bleu_score, blip_bleu_score]
    meteor_scores = [vit_meteor_score, blip_meteor_score]

    data = {
        'Model': ['ViT', 'BLIP'],
        'BLEU Score': [vit_bleu_score, blip_bleu_score],
        'METEOR Score': [vit_meteor_score, blip_meteor_score]
    }

    df = pd.DataFrame(data)

    # Create a table plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Save the table plot as an image file
    table_path = 'static/scores_table.png'
    plt.savefig(table_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar(models, bleu_scores, color=['skyblue', 'orange'])
    plt.ylabel('BLEU Score')
    plt.title('BLEU Scores for ViT and BLIP Models')
    plt.savefig('static/bleu_scores.png')  # Save the bar graph as an image file
    plt.close()  # Close the plot to prevent displaying the plot in Flask app

    plt.figure(figsize=(8, 6))
    plt.bar(models, meteor_scores, color=['skyblue', 'orange'])
    plt.ylabel('METEOR Score')
    plt.title('METEOR Scores for ViT and BLIP Models')
    plt.savefig('static/meteor_scores.png')  # Save the bar graph as an image file
    plt.close()
    
    image_filename = file.filename
    image_path = os.path.join("static", image_filename)
    i_image.save(image_path)

    # Return the evaluation result
    return image_filename, reference_caption, vit_caption,  blip_caption, 'static/bleu_scores.png','static/meteor_scores.png',table_path


if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
