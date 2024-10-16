import streamlit as st  # Import the streamlit library for building the app interface
import base64  # Import the base64 library for encoding images
from loginandsignup import login, signup  # Import the login and signup modules
from chatbot import run_legal_case_chatbot  # Import the function to run the legal case chatbot

def get_base64_encoded_image(image_path):  # Function to encode an image file as base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def main():
    if 'page' not in st.session_state:  # Check if the 'page' key is not in the session state
        st.session_state.page = 'main'  # Set the initial page to 'main'
    if 'logged_in' not in st.session_state:  # Check if the 'logged_in' key is not in the session state
        st.session_state.logged_in = False  # Set the initial logged_in state to False

    if st.session_state.page == 'main':  # If the current page is 'main'
        image_path = 'src/loginandsignup/images/chatbot_pic.png'  # Set the path to the chatbot image
        encoded_image = get_base64_encoded_image(image_path)  # Encode the image as base64

        # Define the CSS styles to set the background image and customize the UI
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}

        [data-testid="stToolbar"] {{
        right: 2rem;
        }}

        .button-container {{
        display: flex;
        justify-content: center;
        position: absolute;
        bottom: 2rem;
        width: 100%;
        }}

        .button-container .stButton {{
        margin: 0 1rem;
        }}
        </style>
        """

        st.markdown(page_bg_img, unsafe_allow_html=True)  # Apply the CSS styles
        st.markdown("<h1 style='text-align: center;'>Hello! I am your legal AI assistant!</h1>", unsafe_allow_html=True)  # Display the heading
        st.markdown("---")  # Display a horizontal line

        container = st.container()  # Create a container for the content
        with container:
            st.markdown('''
<p style='text-align: center;'>
    Our juridical chatbot is an indispensable tool for navigating the complexities of the legal system. Whether you're grappling with contractual disputes, seeking guidance on family law matters, or delving into the intricacies of criminal proceedings, our chatbot stands ready to assist you every step of the way.
</p>
<p style='text-align: center;'>
    Imagine having a knowledgeable legal advisor available at any time, ready to provide answers to your most pressing questions, clarify confusing legal terminology, and offer guidance on the best course of action for your specific circumstances. That's precisely what our chatbot offers.
</p>
<p style='text-align: center;'>
    Here's how it works: You provide detailed information about your case, leaving no stone unturned. The more information you provide, the more accurate and tailored our chatbot's response will be. Once armed with all the pertinent details, our chatbot deliberates much like a human judge, weighing the facts, considering relevant laws and precedents, and ultimately delivering a judgment along with practical advice to help you navigate the legal landscape effectively.
</p>
<p style='text-align: center;'>
    Please note that all personally identifiable information (PII) you provide will be tokenized and cannot be identified in any way, ensuring your privacy and confidentiality throughout the interaction.
</p>
<p style='text-align: center;'>
    But remember, accuracy is paramount. To ensure the best possible outcome, it's crucial to specify every detail of your case, no matter how seemingly insignificant it may seem. With our chatbot's expertise at your disposal, you can approach your legal matters with confidence, knowing you have a reliable ally by your side to guide you through even the most challenging legal challenges.
</p>
''', unsafe_allow_html=True)  # Display the content with HTML formatting

            st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True)  # Add some vertical space

            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])  # Create six columns
            with col2:
                if st.button('Login'):  # If the 'Login' button is clicked
                    st.session_state.page = 'login'  # Set the page to 'login'
                    st.experimental_rerun()  # Re-run the app
            with col5:
                if st.button('Sign Up'):  # If the 'Sign Up' button is clicked
                    st.session_state.page = 'signup'  # Set the page to 'signup'
                    st.experimental_rerun()  # Re-run the app

    elif st.session_state.page == 'login':  # If the current page is 'login'
        image_path = 'src/loginandsignup/images/chatbot_pic.png'  # Set the path to the chatbot image
        encoded_image = get_base64_encoded_image(image_path)  # Encode the image as base64

        # Define the CSS styles to set the background image and customize the UI
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}

        [data-testid="stToolbar"] {{
        right: 2rem;
        }}

        .button-container {{
        display: flex;
        justify-content: center;
        position: absolute;
        bottom: 2rem;
        width: 100%;
        }}

        .button-container .stButton {{
        margin: 0 1rem;
        }}
        </style>
        """

        st.markdown(page_bg_img, unsafe_allow_html=True)  # Apply the CSS styles
        container = st.container()  # Create a container for the content
        with container:
            st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)  # Add some vertical space
            if st.button('Back to Main', key='back_to_main_login'):  # If the 'Back to Main' button is clicked
                st.session_state.page = 'main'  # Set the page to 'main'
                st.experimental_rerun()  # Re-run the app
            else:
                if login.login():  # If the login is successful
                    st.session_state.page = 'chatbot'  # Set the page to 'chatbot'
                    st.experimental_rerun()  # Re-run the app

    elif st.session_state.page == 'chatbot':  # If the current page is 'chatbot'
            run_legal_case_chatbot(st.session_state.user_id)  # Run the legal case chatbot with the user ID

    elif st.session_state.page == 'signup':  # If the current page is 'signup'
        image_path = 'src/loginandsignup/images/chatbot_pic.png'  # Set the path to the chatbot image
        encoded_image = get_base64_encoded_image(image_path)  # Encode the image as base64

        # Define the CSS styles to set the background image and customize the UI
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}

        [data-testid="stToolbar"] {{
        right: 2rem;
        }}

        .button-container {{
        display: flex;
        justify-content: center;
        position: absolute;
        bottom: 2rem;
        width: 100%;
        }}

        .button-container .stButton {{
        margin: 0 1rem;
        }}
        </style>
        """

        st.markdown(page_bg_img, unsafe_allow_html=True)  # Apply the CSS styles
        container = st.container()  # Create a container for the content
        with container:
            st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)  # Add some vertical space
            if st.button('Back to Main', key='back_to_main_signup'):  # If the 'Back to Main' button is clicked
                st.session_state.page = 'main'  # Set the page to 'main'
                st.experimental_rerun()  # Re-run the app
            else:
                signup.sign_up()  # Run the sign_up function

if __name__ == "__main__":
    main()  # Call the main function