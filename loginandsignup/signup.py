import streamlit as st  # Import Streamlit for building the web app
import datetime  # Import datetime for handling dates and times
import re  # Import regex for validating email and username
import psycopg2  # Import psycopg2 for PostgreSQL database connection
import bcrypt  # Import bcrypt for password hashing

# PostgreSQL connection details
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "users"
DB_USER = "postgres"
DB_PASSWORD = "amine"

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    :return: Database connection object
    """
    conn = psycopg2.connect(
        host=DB_HOST,        # Database host address
        port=DB_PORT,        # Database port number
        dbname=DB_NAME,      # Database name
        user=DB_USER,        # Database user
        password=DB_PASSWORD # Database user's password
    )
    return conn

def insert_user(email, username, password):
    """
    Inserts a new user into the database.
    :param email: User's email
    :param username: User's username
    :param password: User's password
    :return: None
    """
    created_on = datetime.datetime.now()  # Get the current date and time

    conn = get_db_connection()  # Establish a database connection
    cur = conn.cursor()  # Create a cursor object
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())  # Hash the password using bcrypt
    cur.execute(
        "INSERT INTO users (username, password, email, created_on) VALUES (%s, %s, %s, %s)",
        (username, hashed_password.decode('utf-8'), email, created_on)  # Execute the SQL statement to insert user
    )
    conn.commit()  # Commit the transaction and persist the inserted row
    cur.close()  # Close the cursor
    conn.close()  # Close the connection

def fetch_users():
    """
    Fetches all users from the database.
    :return: List of users
    """
    conn = get_db_connection()  # Establish a database connection
    cur = conn.cursor()  # Create a cursor object
    cur.execute("SELECT * FROM users")  # Execute the SQL statement to fetch all users
    users = cur.fetchall()  # Fetch all rows from the executed query
    cur.close()  # Close the cursor
    conn.close()  # Close the connection
    return users  # Return the list of users

def get_user_emails():
    """
    Fetches all user emails from the database.
    :return: List of user emails
    """
    conn = get_db_connection()  # Establish a database connection
    cur = conn.cursor()  # Create a cursor object
    cur.execute("SELECT email FROM users")  # Execute the SQL statement to fetch all user emails
    emails = [row[0] for row in cur.fetchall()]  # Extract emails from the fetched rows
    cur.close()  # Close the cursor
    conn.close()  # Close the connection
    return emails  # Return the list of emails

def get_usernames():
    """
    Fetches all usernames from the database.
    :return: List of usernames
    """
    conn = get_db_connection()  # Establish a database connection
    cur = conn.cursor()  # Create a cursor object
    cur.execute("SELECT username FROM users")  # Execute the SQL statement to fetch all usernames
    usernames = [row[0] for row in cur.fetchall()]  # Extract usernames from the fetched rows
    cur.close()  # Close the cursor
    conn.close()  # Close the connection
    return usernames  # Return the list of usernames

def validate_email(email):
    """
    Validates the email format.
    :param email: Email to validate
    :return: True if email is valid, else False
    """
    pattern = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"  # Regex pattern for email validation

    if re.match(pattern, email):  # Check if email matches the pattern
        return True
    return False

def validate_username(username):
    """
    Validates the username format.
    :param username: Username to validate
    :return: True if username is valid, else False
    """
    pattern = "^[a-zA-Z0-9]+$"  # Regex pattern for username validation
    if re.match(pattern, username):  # Check if username matches the pattern
        return True
    return False

def sign_up():
    st.subheader('Sign Up')  # Display a subheader for the sign-up section
    email = st.text_input(':blue[Email]', placeholder='Enter Your Email')  # Input field for email
    username = st.text_input(':blue[Username]', placeholder='Enter Your Username')  # Input field for username
    password1 = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')  # Input field for password
    password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')  # Input field for password confirmation

    if st.button('Sign Up'):  # Button for sign-up
        if email:  # Check if email is provided
            if validate_email(email):  # Validate email format
                if email not in get_user_emails():  # Check if email already exists
                    if validate_username(username):  # Validate username format
                        if username not in get_usernames():  # Check if username already exists
                            if len(username) >= 2:  # Check if username length is at least 2
                                if len(password1) >= 6:  # Check if password length is at least 6
                                    if password1 == password2:  # Check if passwords match
                                        # Add User to DB
                                        insert_user(email, username, password1)  # Insert user into the database
                                        st.success('Account created successfully!!')  # Display success message
                                    else:
                                        st.warning('Passwords Do Not Match')  # Display warning if passwords do not match
                                else:
                                    st.warning('Password is too Short')  # Display warning if password is too short
                            else:
                                st.warning('Username Too short')  # Display warning if username is too short
                        else:
                            st.warning('Username Already Exists')  # Display warning if username already exists
                    else:
                        st.warning('Invalid Username')  # Display warning if username is invalid
                else:
                    st.warning('Email Already exists!!')  # Display warning if email already exists
            else:
                st.warning('Invalid Email')  # Display warning if email is invalid
