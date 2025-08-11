import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import io
import tensorflow as tf
import joblib
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import cv2

# Set page configuration
st.set_page_config(
    page_title="Wildfire Analysis Tool",
    page_icon="🔥",
    layout="wide",
)

# Add custom styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #ff7043;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff5722;
    }
    .reportview-container {
        background-color: #f0f2f6;
    }
    h1 {
        color: #ff5722;
    }
    h2 {
        color: #ff7043;
    }
    h3 {
        color: #ff8a65;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# U-Net Generator for GAN
# ----------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        self.enc1 = self._encoder_block(in_channels, 64, use_batchnorm=False)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        self.enc5 = self._encoder_block(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec5 = self._decoder_block(1024, 512)
        self.dec4 = self._decoder_block(1024, 256)
        self.dec3 = self._decoder_block(512, 128) 
        self.dec2 = self._decoder_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _encoder_block(self, in_channels, out_channels, use_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d5 = self.dec5(torch.cat([b, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        output = self.final(torch.cat([d2, e1], dim=1))
        return output

# Function to load the GAN model
# Improved GAN model loading function
@st.cache_resource
def load_generator_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    
    try:
        # Handle different path formats
        model_path = model_path.strip()
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, device
            
        # Load the model with better error handling
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if we need to adjust keys for compatibility
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            st.warning(f"Standard loading failed: {str(e)}. Trying alternative loading method...")
            
            # Try removing 'module.' prefix if it exists
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            try:
                model.load_state_dict(new_state_dict)
            except Exception as e2:
                st.error(f"Alternative loading also failed: {str(e2)}")
                return None, device
        
        model.eval()
        st.success("GAN model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"Error loading GAN model: {str(e)}")
        return None, device

# Function to process the image and generate post-fire image with improved quality
# Function to process the image and generate post-fire image with improved error handling
def generate_post_fire_image(model, image, device):
    if model is None:
        st.error("GAN model not loaded properly")
        return None
        
    try:
        # Define the same transforms used during training with better interpolation
        transform = transforms.Compose([
            transforms.Resize((512, 512), Image.BICUBIC),  # Use better interpolation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Apply transforms to the input image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate post-fire image
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Convert output tensor to PIL image
        output_image = output_tensor.squeeze(0).cpu()
        # Denormalize
        output_image = (output_image + 1) / 2
        output_image = transforms.ToPILImage()(output_image)
        
        return output_image
    except Exception as e:
        st.error(f"Error generating post-fire image: {str(e)}")
        return None

# ----------------------------
# CNN Classification Functions
# ----------------------------

# Custom function to get model input shape for debugging
def get_model_input_shape(model):
    # Try to extract input shape from the first layer
    try:
        config = model.get_config()
        if 'layers' in config and len(config['layers']) > 0:
            first_layer = config['layers'][0]
            if 'batch_input_shape' in first_layer['config']:
                input_shape = first_layer['config']['batch_input_shape']
                return input_shape[1:]  # Exclude batch dimension
    except:
        pass
    
    # Fallback to a common default if we can't determine automatically
    return (224, 224, 3)

# Function to load the CNN model with debugging
# Improved CNN model loading function
@st.cache_resource
def load_cnn_model(model_path):
    try:
        model_path = model_path.strip()
        if not os.path.exists(model_path):
            st.error(f"CNN model file not found: {model_path}")
            return None
            
        # Try different loading options
        try:
            # First, try standard loading
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e1:
            st.warning(f"Standard loading failed: {str(e1)}. Trying alternative loading method...")
            
            # Try with custom object scope
            try:
                with tf.keras.utils.custom_object_scope({'SamplingLayer': SamplingLayer}):
                    model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e2:
                st.error(f"Alternative loading also failed: {str(e2)}")
                return None
        
        # Debug information
        input_shape = get_model_input_shape(model)
        st.success(f"CNN model loaded successfully! Expected input shape: {input_shape}")
        
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {str(e)}")
        return None

# Function to preprocess image for CNN with correct dimensions
def preprocess_image_for_cnn(image, model):
    # Get the expected input shape directly from the model
    input_shape = None
    
    # First method: try to get from model config if it has get_config method
    try:
        config = model.get_config()
        if 'layers' in config and len(config['layers']) > 0:
            # Try to get from first layer
            first_layer = config['layers'][0]
            if 'batch_input_shape' in first_layer['config']:
                input_shape = first_layer['config']['batch_input_shape'][1:3]
    except:
        pass
    
    # Second method: try to get from input shape attribute
    if input_shape is None:
        try:
            input_shape = model.input_shape[1:3]
        except:
            # Default fallback
            input_shape = (224, 224)
    
    st.info(f"Using target size based on model input: {input_shape}")
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if not already (handles grayscale or RGBA)
    if len(img_array.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:  # Already RGB
        img_rgb = img_array[:, :, :3]  # Just ensure 3 channels
    
    # Resize to the expected input shape
    img_resized = cv2.resize(img_rgb, input_shape)
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    st.info(f"Preprocessed image shape: {img_normalized.shape}")
    
    return img_normalized
# Function to classify image using CNN
def classify_image_with_cnn(model, image):
    # Define class labels
    class_labels = ['fire', 'nofire', 'smoke', 'smokefire']
    
    # Preprocess the image using the model information
    processed_img = preprocess_image_for_cnn(image, model)
    
    if processed_img is None:
        return {
            'class': 'error',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in class_labels}
        }
    
    # Prepare for prediction
    input_array = np.expand_dims(processed_img, axis=0)
    
    # Make prediction with error handling
    try:
        prediction = model.predict(input_array)
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_idx]
        
        # Get confidence score
        confidence = float(prediction[0][predicted_class_idx])
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Input shape: " + str(input_array.shape))
        
        # Try to print model summary for debugging
        try:
            stringio = io.StringIO()
            model.summary(print_fn=lambda x: stringio.write(x + '\n'))
            model_summary = stringio.getvalue()
            st.code(model_summary)
        except:
            pass
            
        return {
            'class': 'error',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in class_labels}
        }
# Function to create annotated image with classification results
def create_annotated_image(image, class_name, confidence):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create a copy for annotations
    annotated_img = img_array.copy()
    
    # Add text annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{class_name} ({confidence:.1f}%)"
    
    # Calculate position (bottom left corner with padding)
    text_x = 10
    text_y = annotated_img.shape[0] - 20
    
    # Create rectangle for text background
    (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
    cv2.rectangle(annotated_img, 
                  (text_x - 5, text_y - text_height - 5),
                  (text_x + text_width + 5, text_y + 5), 
                  (0, 0, 0), -1)
    
    # Add text
    cv2.putText(annotated_img, text, (text_x, text_y), 
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return Image.fromarray(annotated_img)

# Function to display classification results
def display_classification_results(results, image, column):
    with column:
        st.subheader("Classification Results")
        
        # Handle error case
        if results['class'] == 'error':
            st.error("Classification failed. See error message above.")
            return
        
        # Display the predicted class
        class_name = results['class']
        confidence = results['confidence'] * 100
        
        # Map classes to emojis and colors
        class_icons = {
            'fire': '🔥',
            'nofire': '✅',
            'smoke': '💨',
            'smokefire': '🔥💨'
        }
        
        class_colors = {
            'fire': 'red',
            'nofire': 'green',
            'smoke': 'gray',
            'smokefire': 'orange'
        }
        
        icon = class_icons.get(class_name, '❓')
        color = class_colors.get(class_name, 'blue')
        
        # Display the result with styling
        st.markdown(
            f"<h3 style='color: {color};'>{icon} {class_name.upper()} ({confidence:.2f}%)</h3>",
            unsafe_allow_html=True
        )
        
        # Display probabilities as a bar chart
        st.write("### Class Probabilities")
        probs = results['probabilities']
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = list(probs.keys())
        values = list(probs.values())
        
        # Use class-specific colors for the bars
        colors = [class_colors.get(cls, 'blue') for cls in classes]
        bars = ax.bar(classes, [v * 100 for v in values], color=colors)
        
        # Add labels and formatting
        ax.set_ylabel('Probability (%)')
        ax.set_ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Create and display the annotated image
        annotated_img = create_annotated_image(image, class_name, confidence)
        st.subheader("Annotated Image")
        st.image(annotated_img, use_column_width=True)
        
        # Add download button for the annotated image
        buf = BytesIO()
        annotated_img.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label="Download Annotated Image",
            data=buf,
            file_name=f"classified_{class_name}.png",
            mime="image/png"
        )

# ----------------------------
# VAE Anomaly Detection Functions
# ----------------------------

# Add this custom SamplingLayer to your Streamlit app
# It should match the same layer definition from your training code
class SamplingLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Register the custom layer globally to ensure it's available during model loading
tf.keras.utils.get_custom_objects().update({'SamplingLayer': SamplingLayer})

# Improved function to load VAE models with better error handling and file checking
@st.cache_resource
def load_vae_models(model_dir):
    """
    Load VAE models from specified directory with enhanced error handling and debugging
    """
    try:
        # Check if directory exists
        if not os.path.exists(model_dir):
            st.error(f"Directory not found: {model_dir}")
            return None, None, None, None
        
        # List all files in the directory for debugging
        all_files = os.listdir(model_dir)
        st.success(f"Found {len(all_files)} files in model directory")
        
        # Try to find and load scaler file
        scaler = None
        scaler_files = ['thermal_scaler.pkl', 'scaler.pkl', 'thermal_scaler', 'scaler']
        
        for scaler_file in scaler_files:
            scaler_path = os.path.join(model_dir, scaler_file)
            if os.path.exists(scaler_path):
                try:
                    st.info(f"Loading scaler from {scaler_file}")
                    scaler = joblib.load(scaler_path)
                    st.success(f"Scaler loaded successfully")
                    break
                except Exception as e:
                    st.warning(f"Failed to load {scaler_file}: {str(e)}")
        
        # If no scaler found, create a default one
        if scaler is None:
            st.warning("No scaler found. Creating a default StandardScaler.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        # Try to load encoder model with multiple approaches
        encoder = None
        encoder_files = ['thermal_encoder.keras', 'thermal_encoder.h5', 'encoder.keras', 'encoder.h5', 'encoder']
        
        for encoder_file in encoder_files:
            encoder_path = os.path.join(model_dir, encoder_file)
            if os.path.exists(encoder_path):
                st.info(f"Trying to load encoder from {encoder_file}")
                
                # Try multiple loading approaches
                try:
                    # Standard loading
                    encoder = tf.keras.models.load_model(
                        encoder_path, 
                        custom_objects={'SamplingLayer': SamplingLayer}
                    )
                    st.success("Encoder loaded successfully")
                    break
                except Exception as e1:
                    st.warning(f"Standard loading failed: {str(e1)}")
                    
                    # Try alternative loading
                    try:
                        with tf.keras.utils.custom_object_scope({'SamplingLayer': SamplingLayer}):
                            encoder = tf.keras.models.load_model(encoder_path)
                        st.success("Encoder loaded with custom object scope")
                        break
                    except Exception as e2:
                        st.warning(f"Alternative loading also failed: {str(e2)}")
        
        if encoder is None:
            st.error("Could not load encoder model. VAE functionality will not work.")
            return None, None, None, None
        
        # Similar approach for decoder
        decoder = None
        decoder_files = ['thermal_decoder.keras', 'thermal_decoder.h5', 'decoder.keras', 'decoder.h5', 'decoder']
        
        for decoder_file in decoder_files:
            decoder_path = os.path.join(model_dir, decoder_file)
            if os.path.exists(decoder_path):
                st.info(f"Trying to load decoder from {decoder_file}")
                
                # Try multiple loading approaches
                try:
                    # Standard loading
                    decoder = tf.keras.models.load_model(decoder_path)
                    st.success("Decoder loaded successfully")
                    break
                except Exception as e1:
                    st.warning(f"Standard loading failed: {str(e1)}")
                    
                    # Try alternative loading
                    try:
                        with tf.keras.utils.custom_object_scope({'SamplingLayer': SamplingLayer}):
                            decoder = tf.keras.models.load_model(decoder_path)
                        st.success("Decoder loaded with custom object scope")
                        break
                    except Exception as e2:
                        st.warning(f"Alternative loading also failed: {str(e2)}")
        
        if decoder is None:
            st.error("Could not load decoder model. VAE functionality will not work.")
            return None, None, None, None
        
        # Try to load threshold value
        threshold = None
        threshold_files = ['anomaly_threshold.npy', 'threshold.npy']
        
        for threshold_file in threshold_files:
            threshold_path = os.path.join(model_dir, threshold_file)
            if os.path.exists(threshold_path):
                try:
                    threshold = float(np.load(threshold_path))
                    st.success(f"Threshold loaded: {threshold}")
                    break
                except Exception as e:
                    st.warning(f"Failed to load {threshold_file}: {str(e)}")
        
        # If no threshold file exists, use default
        if threshold is None:
            st.warning("No threshold file found. Using default value of 0.1")
            threshold = 0.1
        
        return scaler, encoder, decoder, threshold
    
    except Exception as e:
        st.error(f"Error in load_vae_models: {str(e)}")
        return None, None, None, None

# Function to process CSV data and detect anomalies
def predict_anomaly_from_csv(df, models):
    """
    Function to predict anomalies in CSV data with improved error handling
    """
    scaler, encoder, decoder, threshold = models
    
    if encoder is None or decoder is None:
        st.error("VAE models not properly loaded. Cannot perform anomaly detection.")
        return None
    
    # Prepare data - assuming all numeric columns are features
    # Filter only numeric columns
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Check if we have any numeric columns
        if numeric_df.empty:
            st.error("No numeric columns found in the CSV file. VAE requires numeric data.")
            return None
        
        # Show which columns will be used for anomaly detection
        st.info(f"Using {len(numeric_df.columns)} numeric columns for anomaly detection: {', '.join(numeric_df.columns)}")
        
        # Features
        features = numeric_df.values
        
        # Scale the input using the loaded scaler
        try:
            scaled_input = scaler.transform(features)
        except Exception as e:
            st.error(f"Error scaling input data: {str(e)}")
            st.warning("Fitting a new scaler to the data. This may affect anomaly detection accuracy.")
            
            # Try to fit a new scaler if the existing one doesn't work
            from sklearn.preprocessing import StandardScaler
            new_scaler = StandardScaler()
            scaled_input = new_scaler.fit_transform(features)
        
        # Get latent representation
        try:
            # Check the encoder's expected input shape
            encoder_input_shape = encoder.input_shape
            st.info(f"Encoder expects input shape: {encoder_input_shape}")
            st.info(f"Provided data shape: {scaled_input.shape}")
            
            # Make prediction
            encoder_outputs = encoder.predict(scaled_input, verbose=0)
            
            # Handle different possible encoder output structures
            if isinstance(encoder_outputs, list) and len(encoder_outputs) >= 3:
                z_mean_input, z_log_var_input, z_input = encoder_outputs
            elif isinstance(encoder_outputs, list) and len(encoder_outputs) == 2:
                z_mean_input, z_log_var_input = encoder_outputs
                # Apply sampling manually
                batch = z_mean_input.shape[0]
                dim = z_mean_input.shape[1]
                epsilon = np.random.normal(size=(batch, dim))
                z_input = z_mean_input + np.exp(0.5 * z_log_var_input) * epsilon
            else:
                # Assume single output is the latent representation
                z_input = encoder_outputs
                z_mean_input = encoder_outputs
                z_log_var_input = np.zeros_like(z_mean_input)
        except Exception as e:
            st.error(f"Error in encoder prediction: {str(e)}")
            return None
        
        # Get reconstruction
        try:
            # Check the decoder input shape
            decoder_input_shape = decoder.input_shape
            st.info(f"Decoder expects input shape: {decoder_input_shape}")
            st.info(f"Latent representation shape: {z_input.shape}")
            
            reconstructed_input = decoder.predict(z_input, verbose=0)
        except Exception as e:
            st.error(f"Error in decoder prediction: {str(e)}")
            return None
        
        # Calculate reconstruction error for each row
        mse_input = np.mean(np.square(scaled_input - reconstructed_input), axis=1)
        
        # Determine if each row is an anomaly
        is_anomaly = mse_input > threshold
        
        # Calculate anomaly scores
        anomaly_scores = mse_input / threshold
        
        # Return results
        return {
            'reconstruction_errors': mse_input,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'anomaly_scores': anomaly_scores,
            'reconstructed_data': reconstructed_input,
            'latent_representations': z_input,
            'z_mean': z_mean_input,
            'z_log_var': z_log_var_input
        }
    except Exception as e:
        st.error(f"Error in anomaly prediction: {str(e)}")
        return None

# Function to display CSV anomaly detection results
def display_csv_anomaly_results(results, df):
    # Set pandas option to allow displaying more cells
    pd.set_option("styler.render.max_elements", 1500000)  # Set higher than your cell count
    
    # Display summary statistics
    anomaly_count = np.sum(results['is_anomaly'])
    total_count = len(results['is_anomaly'])
    anomaly_percentage = (anomaly_count / total_count) * 100
    
    st.subheader("Anomaly Detection Results")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", total_count)
    col2.metric("Anomalies Detected", anomaly_count)
    col3.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")
    
    # Create gauge chart for anomaly percentage
    fig_gauge = plt.figure(figsize=(6, 3))
    ax = fig_gauge.add_subplot(111)
    gauge_colors = ['green', 'yellow', 'orange', 'red']
    thresholds = [0, 10, 30, 60, 100]
    gauge_color = gauge_colors[0]
    for i in range(1, len(thresholds)):
        if anomaly_percentage >= thresholds[i-1] and anomaly_percentage <= thresholds[i]:
            gauge_color = gauge_colors[i-1]
    ax.barh(1, 100, color='lightgray', height=0.5)
    ax.barh(1, anomaly_percentage, color=gauge_color, height=0.5)
    ax.text(50, 1, f"{anomaly_percentage:.2f}% Anomalies", ha='center', va='center', fontweight='bold')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 100)
    ax.set_frame_on(False)
    st.pyplot(fig_gauge)
    
    # Add anomaly flags to dataframe
    result_df = df.copy()
    result_df['Reconstruction_Error'] = results['reconstruction_errors']
    result_df['Anomaly_Score'] = results['anomaly_scores']
    result_df['Is_Anomaly'] = results['is_anomaly']
    
    # For large datasets, show only a sample to avoid display issues
    if len(result_df) > 10000:
        st.warning(f"Dataset is large ({len(result_df)} rows). Showing only anomalies and a sample of normal records.")
        
        # Get all anomalies
        anomalies_df = result_df[result_df['Is_Anomaly']]
        
        # Get a sample of non-anomalies (maximum 1000)
        non_anomalies = result_df[~result_df['Is_Anomaly']].sample(min(1000, len(result_df[~result_df['Is_Anomaly']])))
        
        # Combine and sort
        display_df = pd.concat([anomalies_df, non_anomalies]).sort_index()
        
        # Apply styling to highlight anomalies
        def highlight_anomalies(s):
            is_anomaly = s['Is_Anomaly']
            return ['background-color: rgba(255, 0, 0, 0.2)' if is_anomaly else '' for _ in s]
        
        styled_df = display_df.style.apply(highlight_anomalies, axis=1)
        st.dataframe(styled_df)
    else:
        # Apply styling to highlight anomalies
        def highlight_anomalies(s):
            is_anomaly = s['Is_Anomaly']
            return ['background-color: rgba(255, 0, 0, 0.2)' if is_anomaly else '' for _ in s]
        
        styled_df = result_df.style.apply(highlight_anomalies, axis=1)
        st.dataframe(styled_df)
    
    
    # Plot reconstruction loss distribution
    st.subheader("Reconstruction Loss Distribution")
    fig_loss = plt.figure(figsize=(10, 6))
    plt.hist(results['reconstruction_errors'], bins=50, alpha=0.7, color='blue', label='All samples')
    
    # Add anomalies in a different color
    if anomaly_count > 0:
        plt.hist(results['reconstruction_errors'][results['is_anomaly']], 
                 bins=20, alpha=0.7, color='red', label='Anomalies')
        
    plt.axvline(x=results['threshold'], color='red', linestyle='--', 
                label=f'Threshold: {results["threshold"]:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_loss)
    
    # Plot latent space representation (first 2 dimensions) if available
    if results['latent_representations'].shape[1] >= 2:
        st.subheader("Latent Space Representation")
        fig_latent = plt.figure(figsize=(10, 8))
        
        # Use different colors for normal vs anomaly points
        normal_mask = ~results['is_anomaly']
        anomaly_mask = results['is_anomaly']
        
        plt.scatter(
            results['latent_representations'][normal_mask, 0],
            results['latent_representations'][normal_mask, 1],
            c='blue', alpha=0.5, label='Normal'
        )
        
        if np.any(anomaly_mask):
            plt.scatter(
                results['latent_representations'][anomaly_mask, 0],
                results['latent_representations'][anomaly_mask, 1],
                c='red', alpha=0.7, label='Anomaly'
            )
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('2D Projection of Latent Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_latent)
        
        # Add download buttons for results
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="Download Results as CSV",
            data=csv_buffer,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )

# VAE-Anomaly Detection UI
def vae_anomaly_detection():
    st.header("VAE-Anomaly Detection")
    
    # Create info box with explanation
    st.info("""
    This module uses a Variational Autoencoder (VAE) to detect anomalies in your CSV data.
    Upload a CSV file containing numeric features, and the model will identify potential anomalies.
    
    **How it works:** The VAE learns the normal pattern of your data and flags samples that deviate significantly.
    """)
    
    # Default path for VAE models - updated to use the correct path
    default_vae_dir = r"D:\WILDFIRE_GENAI\model"
    
    # Allow custom path input
    vae_dir = st.text_input("VAE Model Directory", value=default_vae_dir)
    
    # Check if directory exists
    if not os.path.exists(vae_dir):
        st.warning(f"Directory {vae_dir} does not exist. Please check the path.")
    else:
        # Display files in directory
        try:
            model_files = os.listdir(vae_dir)
            st.write("Files in directory:", model_files)
        except Exception as e:
            st.error(f"Error listing directory contents: {str(e)}")
        
        # Load VAE models with improved error handling
        with st.spinner("Loading VAE models..."):
            vae_models = load_vae_models(vae_dir)
        
        # File uploader for CSV data
        uploaded_csv = st.file_uploader("Upload CSV file for anomaly detection", type=['csv'])
        
        if uploaded_csv is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_csv)
                st.write("CSV Data Preview:")
                st.dataframe(df.head())
                
                # Check if we have numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    st.error("No numeric columns found in the CSV file. VAE requires numeric data.")
                else:
                    # Run anomaly detection
                    with st.spinner("Detecting anomalies..."):
                        anomaly_results = predict_anomaly_from_csv(df, vae_models)
                    
                    if anomaly_results is not None:
                        # Display results
                        display_csv_anomaly_results(anomaly_results, df)
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")

# ----------------------------
# Main App Interface
# ----------------------------
def main():
    st.title("🔥 Wildfire Analysis Tool")
    st.write("An all-in-one tool for wildfire detection, simulation, and analysis")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["GAN Fire Simulation", "CNN Classification", "VAE Anomaly Detection"])
    
    with tab1:
        st.header("GAN Fire Simulation")
        st.write("Generate post-fire scenarios from pre-fire images")
        
        # Default path for the GAN model - update with your new path
        default_model_path = r"D:\WILDFIRE_GENAI\generator_best (2).pth"
        
        # Allow custom path input
        gan_model_path = st.text_input("GAN Model Path", value=default_model_path)
        
        # Load GAN model with better error handling
        with st.spinner("Loading GAN model..."):
            gan_model, device = load_generator_model(gan_model_path)
        
        if gan_model is not None:
            # File uploader for pre-fire images
            uploaded_file = st.file_uploader("Upload a pre-fire image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                
                # Create columns for before/after comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    # Resize to 512x512 for display
                    display_img = image.resize((512, 512), Image.LANCZOS)
                    st.image(display_img, width=512)
                
                # Generate button
                if st.button("Generate Post-Fire Scenario"):
                    with st.spinner("Generating..."):
                        try:
                            # Generate post-fire image
                            post_fire_image = generate_post_fire_image(gan_model, image, device)
                            
                            # Display generated image
                            with col2:
                                st.subheader("Generated Post-Fire Scenario")
                                # Ensure the output is 512x512
                                post_fire_display = post_fire_image.resize((512, 512), Image.LANCZOS)
                                st.image(post_fire_display, width=512)
                                
                                # Add download button
                                buf = BytesIO()
                                post_fire_image.save(buf, format="PNG")
                                buf.seek(0)
                                
                                st.download_button(
                                    label="Download Generated Image",
                                    data=buf,
                                    file_name="post_fire_scenario.png",
                                    mime="image/png"
                                )
                        except Exception as e:
                            st.error(f"Error generating post-fire image: {str(e)}")
    
    with tab2:
        st.header("CNN Classification")
        st.write("Classify wildfire images into fire, smoke, both, or none")
        
        # Default path for CNN model - update with your new path
        default_cnn_path = r"D:\WILDFIRE_GENAI\best_fire_smoke_cnn.h5"
        
        # Allow custom path input
        cnn_model_path = st.text_input("CNN Model Path", value=default_cnn_path)
        
        # Load CNN model with better error handling
        with st.spinner("Loading CNN model..."):
            cnn_model = load_cnn_model(cnn_model_path)
        
        if cnn_model is not None:
            # File uploader for classification
            uploaded_file = st.file_uploader("Upload an image for classification", type=['jpg', 'jpeg', 'png'], key="cnn_uploader")
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                
                # Create columns for image and results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Image to Classify")
                    st.image(image, use_column_width=True)
                
                # Classify button
                if st.button("Classify Image"):
                    with st.spinner("Classifying..."):
                        try:
                            # Classify image
                            classification_results = classify_image_with_cnn(cnn_model, image)
                            
                            # Display results
                            display_classification_results(classification_results, image, col2)
                        except Exception as e:
                            st.error(f"Error during classification: {str(e)}")
    
    with tab3:
        st.header("VAE-Anomaly Detection")
        
        # Create info box with explanation
        st.info("""
        This module uses a Variational Autoencoder (VAE) to detect anomalies in your CSV data.
        Upload a CSV file containing numeric features, and the model will identify potential anomalies.
        
        **How it works:** The VAE learns the normal pattern of your data and flags samples that deviate significantly.
        """)
        
        # Default path for VAE models
        default_vae_dir = r"E:\SUBASHREE_GENAI\WILDFIRE_ANALYSIS\thermal_anomaly_detection_outputs\model"
        
        # Allow custom path input
        vae_dir = st.text_input("VAE Model Directory", value=default_vae_dir)
        
        # Check if directory exists with better error handling
        if not os.path.exists(vae_dir):
            st.warning(f"Directory {vae_dir} does not exist. Please check the path.")
        else:
            # Display files in directory for debugging
            try:
                model_files = os.listdir(vae_dir)
                with st.expander("Files in directory"):
                    st.write(model_files)
            except Exception as e:
                st.error(f"Error listing directory contents: {str(e)}")
            
            # Load VAE models with improved error handling
            with st.spinner("Loading VAE models..."):
                vae_models = load_vae_models(vae_dir)
            
            # File uploader for CSV data
            uploaded_csv = st.file_uploader("Upload CSV file for anomaly detection", type=['csv'])
            
            if uploaded_csv is not None:
                try:
                    # Read the CSV file
                    df = pd.read_csv(uploaded_csv)
                    
                    # Show data preview in an expander
                    with st.expander("CSV Data Preview"):
                        st.dataframe(df.head())
                    
                    # Check if we have numeric columns
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_columns:
                        st.error("No numeric columns found in the CSV file. VAE requires numeric data.")
                    else:
                        # Run anomaly detection
                        if st.button("Detect Anomalies"):
                            with st.spinner("Detecting anomalies..."):
                                try:
                                    anomaly_results = predict_anomaly_from_csv(df, vae_models)
                                    
                                    if anomaly_results is not None:
                                        # Display results
                                        display_csv_anomaly_results(anomaly_results, df)
                                except Exception as e:
                                    st.error(f"Error during anomaly detection: {str(e)}")
                                    st.info("Try uploading a different CSV or check if the data format matches what the model expects.")
                except Exception as e:
                    st.error(f"Error processing CSV file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2023 Wildfire Analysis Tool | Created for research purposes")

if __name__ == "__main__":
    main()
