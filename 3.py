import streamlit as st
from PIL import Image as PILImage  # Avoid name conflict with the Image class
import numpy as np
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('amcc.keras')
    return model

def preprocessing_image(image):
    target_size = (64, 64)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32')
    return image_array

def predict(model, image):
    return model.predict(image, batch_size=1)

def interpret_prediction(prediction):
    if prediction.shape[-1] == 1:
        score = prediction[0][0]
        predicted_class = 0 if score <= 0.5 else 1
        confidence_scores = [score, 1 - score, 0]
    else:
        confidence_scores = prediction[0]
        predicted_class = np.argmax(confidence_scores)
    return predicted_class, confidence_scores

def main():
    st.set_page_config(
        page_title='Cat or Dog Classifier',
        layout='centered'
    )

    st.title('Cat or Dog Classifier')

    try:
        model = load_model()
        st.sidebar.write('Hello Pet Lovers !')
        st.sidebar.markdown('# How to use')  # Use Markdown to increase the font size
        st.sidebar.write('- Upload your pet image')
        st.sidebar.write('- CLick classify button')
        st.sidebar.write('- Wait for the result')


    except Exception as err:
        st.error(f'Error: {str(err)}')
        return
    
    uploader = st.file_uploader('Choose an Image', type=['png', 'jpg', 'jpeg'])

    if uploader is not None:
        try:
            col1, col2 = st.columns([2, 1])
            with col1:
                image = PILImage.open(uploader)
                st.image(image, caption='Image', use_column_width=True)
            
            with col2:
                if st.button('Classify', use_container_width=True):
                    with st.spinner('Classifying'):
                        processed_image = preprocessing_image(image)
                        prediction = predict(model, processed_image)
                        predicted_class, confidence_scores = interpret_prediction(prediction)
                        class_names = ['Cat', 'Dog']
                        result = class_names[predicted_class]
                        st.success(f'Image: {result.capitalize()}')
                        st.write('Scores:')
                        st.write(f'Dog: {confidence_scores[0] * 100:.2f}%')
                        st.write(f'Cat: {confidence_scores[1] * 100:.2f}%')

        except Exception as err:
            st.error(f'Error: {str(err)}')
            st.write('Choose a correct file')
            st.write(f'Error: {str(err)}')

if __name__ == '__main__':
    main()
