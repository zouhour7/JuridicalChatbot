import streamlit as st # Library for interface interactions
import psycopg2  # PostgreSQL database adapter for Python
import bcrypt  # Library for hashing passwords
import time # Library to control application time/sleep

# PostgreSQL connection details
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "users"
DB_USER = "postgres"
DB_PASSWORD = "amine"

def get_db_connection():
    # Establishes a connection to the PostgreSQL database.
    # :return: Database connection object
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

def authenticate_user(username, password):
    # Establish a database connection
    conn = get_db_connection()
    # creating a cursor object on the database we connected on
    cur = conn.cursor()

    # Execute a SQL query to retrieve the user's ID and hashed password
    cur.execute("SELECT id, password FROM users WHERE username = %s", (username,))
    # Retrieve the first row
    result = cur.fetchone()

    # Close the database cursor and connection
    cur.close()
    conn.close()

    # If a user is found (result isn't null)
    if result:
        user_id, hashed_password = result

        # Check if the provided password matches the stored hashed password
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            # If the passwords match, return True and the user's ID
            return True, user_id
    # If no user is found or the password doesn't match, return False and None
    return False, None

def login():
    st.subheader('Login')
    username = st.text_input(':blue[Username]', placeholder='Enter Your Username')
    password = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')

    # If the user clicks the 'Login' button
    if st.button('Login'):
        # Check if both username and password are provided
        if username and password:
            # Call the authenticate_user function to authenticate the user
            is_authenticated, user_id = authenticate_user(username, password)

            # If the user is authenticated
            if is_authenticated:
                st.success('Logged in successfully!')
                st.session_state.user_id = user_id
                st.session_state.logged_in = True  # Set logged_in to True
                time.sleep(2)  # Wait for 2 seconds
                return True
            else:
                st.error('Invalid username or password')
    return False

