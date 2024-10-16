# Import the Streamlit library, used for creating web applications
import streamlit as st
# Importing AIMessage and HumanMessage from langchain.schema, which are message classes for AI and human interactions
from langchain.schema import AIMessage, HumanMessage
# Importing ContextualCompressionRetriever from langchain.retrievers, which are used for retrieving compressed context data
from langchain.retrievers import ContextualCompressionRetriever
# Importing LLMChainExtractor from langchain.retrievers.document_compressors for extracting chains of language model responses
from langchain.retrievers.document_compressors import LLMChainExtractor
# Importing PromptTemplate from langchain.prompts used for creating prompt templates
from langchain.prompts import PromptTemplate
# Importing LLMChain from langchain.chains used for creating chains of language model calls
from langchain.chains import LLMChain
# Importing HuggingFaceEmbeddings from langchain_community.embeddings for embedding texts using Hugging Face models
from langchain_community.embeddings import HuggingFaceEmbeddings
# Importing RunnableLambda from langchain_core.runnables, which allows running lambda functions within the LangChain framework
from langchain_core.runnables import RunnableLambda
# Importing FAISS from langchain.vectorstores, which is a library for efficient similarity search and clustering of dense vectors
from langchain.vectorstores import FAISS
# Importing ChatOllama from langchain_community.chat_models which allows access to chat models 
from langchain_community.chat_models import ChatOllama
# Importing regular expression module for regex operations
import re
# Importing os module for operating system dependent functionality
import os
# Importing uuid module for generating universally unique identifiers
import uuid
# Importing psycopg2 module for interacting with PostgreSQL databases
import psycopg2
# Importing json module for working with JSON data
import json
# Importing DistilBertForTokenClassification and AutoTokenizer from transformers library, which are model and tokenizer classes for token classification using DistilBERT
from transformers import DistilBertForTokenClassification, AutoTokenizer
# Database connection configuration variables
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "users"
DB_USER = "postgres"
DB_PASSWORD = "amine"
# Define a function for running a legal case chatbot
def run_legal_case_chatbot(user_id):
    # Clear the Streamlit app's current content
    st.empty()

    # Define a nested function to retrieve opinions based on a query and a document embedding retriever (Demb)
    def retrieve_opinion(query, Demb):
        # Create an instance of LLMChainExtractor using the language model stored in Streamlit's session state
        compressor = LLMChainExtractor.from_llm(st.session_state.llm)

        # Create a ContextualCompressionRetriever with the compressor and a retriever derived from the document embeddings
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=Demb.as_retriever(search_kwargs={'k': 2})  # Retrieves top 2 documents
        )

        # Retrieve relevant documents based on the query with a score threshold of 0.9
        relevant_documents = compression_retriever.get_relevant_documents(str(query), score_threshold=0.9)
        
        # Print the retrieved documents for debugging purposes
        print(relevant_documents)
        print('**************************')

        # Initialize an empty list to store the opinions
        opinions = []

        # Loop through each retrieved document
        for doc in relevant_documents:
            # Extract metadata from the document
            metadata = doc.metadata
            
            # Get the source field from the metadata
            source = metadata['source']
            
            # Search for a case identifier pattern (e.g., 'c1') in the source string
            match = re.search(r'c\d+', source)
            
            # If a case identifier is found
            if match:
                # Extract the case identifier
                case_identifier = match.group(0)
                
                # Construct the path to the opinion text file based on the case identifier
                opinion_path = os.path.join(os.path.dirname(source), case_identifier + 'opinion.txt')
                
                # If the opinion text file exists
                if os.path.exists(opinion_path):
                    # Open and read the opinion text file
                    with open(opinion_path, 'r', encoding="utf-8") as opinion_file:
                        opinion = opinion_file.read()
                    
                    # Append the opinion to the list of opinions
                    opinions.append(opinion)

        # Print the collected opinions for debugging purposes
        print(opinions)
        print('**************************')
        # Return the list of opinions
        return opinions


    # Define a function to retrieve law documents based on a query and a law document embedding retriever (Lemb)
    def retrieve_Law(query, Lemb):
        # Create an instance of LLMChainExtractor using the language model stored in Streamlit's session state
        compressor = LLMChainExtractor.from_llm(st.session_state.llm)

        # Create a ContextualCompressionRetriever with the compressor and a retriever derived from the law document embeddings
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=Lemb.as_retriever(search_kwargs={"k": 3})  # Retrieves top 3 law documents
        )

        # Retrieve relevant law documents based on the query
        relevant_Law = compression_retriever.get_relevant_documents(str(query))
        
        # Print the retrieved law documents for debugging purposes
        print(relevant_Law)
        print('**************************')

        # Return the relevant law documents
        return relevant_Law


    # Define a function to create a RAG (Retrieve, Answer, Generate) chain for generating legal opinions
    def create_rag_chain(op, Law, chat_history):
        # Define a prompt template with placeholders for context, California laws, chat history, and user's question
        prompt_template = """Instruction: As a juridical AI specializing in Californian laws, I am capable of providing legal advice based on the information I retrieve and the user's question.

        Context: {context}

        California Laws: {Law}

        Chat History: {chat_history}

        User's Question: {question}

        If the User's Question isn't about the legal field or about a juridical case, I will reply with: "I am sorry, but I can not answer questions outside the legal field."

        My task is to provide a well-structured and understandable legal advice that directly addresses the User's Question. I will avoid phrases that imply uncertainty or doubt unless it is required.
        
        I will never mention the names mentioned in the User's Question. It is prohibited.

        I will use personal pronouns (you, your, etc.) to address the user.

        I will remember to maintain a conversational tone without mentioning their name and encourage the user to consult with a professional lawyer for further assistance without mentioning their name.

        I will not mention any personally identifiable information (PII) mentioned in the User's Question, including names.

        I will use neutral terms such as "the defendant" or "the suspect" when referring to individuals mentioned in the question.

        I will maintain engagement by acknowledging the user's input and responding in a way that fosters continued conversation without mentioning their name.

        I will show empathy when appropriate, acknowledging the user's feelings or concerns without mentioning their name.

        If the user greets me, I will only say "Hello" without mentioning their name.

        Response:

        """

        
        # Create a prompt template using the defined template and input variables
        prompt = PromptTemplate(
            input_variables=["context", "Law", "chat_history", "question"],
            template=prompt_template,
        )

        # Create an instance of LLMChain with the language model stored in Streamlit's session state and the prompt template
        llm_chain = LLMChain(llm=st.session_state.llm, prompt=prompt, return_final_only=True)
        
        # Define a lambda function to provide input data to the RAG chain
        runnable_lambda = RunnableLambda(lambda question: {"context": op, "Law": Law, "question": question, "chat_history": chat_history})
        
        # Return the RAG chain, which retrieves, answers, and generates legal opinions
        return runnable_lambda | llm_chain

    # Define a function to load conversation messages from the database based on the session ID
    def load_conversation(session_id):
        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Create a cursor object to execute SQL queries
        cur = conn.cursor()
        
        # Execute a SQL query to select messages for the specified session ID, ordered by creation time
        cur.execute("""
            SELECT sender, message_text
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at
        """, (session_id,)) #the , after session_id is for the tuple to be actually considered a TUPLE
        
        # Fetch all the messages from the cursor
        messages = cur.fetchall()
        
        # Close the cursor and the database connection
        cur.close()
        conn.close()

        # Initialize an empty list to store chat history messages
        chat_history = []

        # Iterate over the fetched messages
        for sender, message_text in messages:
            # Determine the type of message (user or AI) and append it to the chat history list accordingly
            if sender == 'user':
                chat_history.append(HumanMessage(content=message_text))
            elif sender == 'ai':
                chat_history.append(AIMessage(content=message_text))
        
        # Return the chat history
        return chat_history

    # Define a function to save a message to the database
    def save_message(session_id, sender, message_text):
        # Establish a connection to the PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Create a cursor object to execute SQL queries
        cur = conn.cursor()
        
        # Execute a SQL query to insert the message into the database
        cur.execute("""
            INSERT INTO messages (user_id, session_id, sender, message_text)
            VALUES (%s, %s, %s, %s)
        """, (user_id, session_id, sender, message_text))
        
        # Commit the transaction
        conn.commit()
        
        # Close the cursor and the database connection
        cur.close()
        conn.close()

    # Define a function to generate a conversation name with the language model (llm) based on the chat history
    def generate_conversation_name_with_llm(session_id, chat_history, llm):
        # Define a prompt template for generating a conversation title based on the chat history
        prompt_template = """
        Instruction: I am a title generator for legal chatbot conversations.
        Based on the following chat history, i generate a concise and descriptive title for the conversation:
        {chat_history}
        The conversation title should have a maximum of 10 words.
        Conversation Title:
        """
        
        # Create a prompt template using the defined template and input variables
        prompt = PromptTemplate(
            input_variables=["chat_history"],
            template=prompt_template,
        )
        
        # Convert the chat history to a formatted string
        chat_history_str = ""
        for message in chat_history:
            if isinstance(message, HumanMessage):
                chat_history_str += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                chat_history_str += f"AI: {message.content}\n"
        
        # Create an instance of LLMChain with the language model (llm) and the prompt template
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the LLMChain to generate a conversation name based on the chat history
        conversation_name = chain.run(chat_history=chat_history_str)
        
        # Strip any leading or trailing whitespace from the generated conversation name
        st.session_state.conversation_names[session_id] = conversation_name.strip()
        
        # Return the generated conversation name
        return conversation_name.strip()


    # Define a function to tokenize personally identifiable information (PII) data using a model and tokenizer
    def Tokenize_PII_data(model, tokenizer, input_text):
        # Load the JSON file containing PII labels and their corresponding IDs
        with open('C:\\Users\\amine\\chatbot\\data\\data.json', 'r') as f:
            data = json.load(f)
        
        # Split the input sentence into words
        words = input_text.split()

        # Remove duplicates based on the JSON file
        filtered_words = [word for word in words if word.lower() not in data]

        # Join the words back into a sentence
        filtered_sentence = ' '.join(filtered_words)

        # Tokenize the filtered sentence using the provided tokenizer
        # tensors are returned and they should be pytorch tensors, truncation to remove anything after 512, padding to add values if it is less than 512
        # A tensor is a multi-dimensional array of numerical values. It represents the relations between vectors
        inputs = tokenizer(filtered_sentence, return_tensors="pt", truncation=True, padding=True)

        # Pass the tokenized input through the model to obtain logits
        # the tokenizer returns tensors that can only be read by their respective model, therefore we need to apply the model on these tensors and apply the logits function to obtain them
        logits = model(**inputs).logits

        # Convert the logits to predicted token IDs
        predicted_token_ids = logits.argmax(-1).numpy().tolist()[0]

        # Map token IDs to labels using the data dictionary
        id_to_label = {id: label for label, id in data.items()}

        # Map predicted token IDs to labels
        predicted_labels = [id_to_label[id] for id in predicted_token_ids]

        # Remove special tokens at the end of the sequence
        predicted_labels = predicted_labels[:-2]

        # Remove 'B-' prefix from each label
        predicted_labels = [label.replace('B-', '') for label in predicted_labels]

        # Tokenize the filtered sentence using the provided tokenizer
        tokens = tokenizer.tokenize(filtered_sentence)

        # Ensure the length of tokens and predicted labels match
        assert len(tokens) == len(predicted_labels), "Tokens and labels length must be the same"

        # Define labels to ignore during processing
        ignore_labels = [
            'BUILDINGNUMBER', 'CITY', 'SEX', 'STATE', 'GENDER', 'COUNTY', 'ZIPCODE', 'JOBTYPE', 'JOBTITLE',
            'B-ZIPCODE', 'I-ZIPCODE', 'B-BUILDINGNUMBER', 'I-BUILDINGNUMBER', 'B-CITY', 'B-SEX', 'B-STATE',
            'B-GENDER', 'B-COUNTY', 'B-JOBTYPE', 'B-JOBTITLE', 'I-CITY', 'I-SEX', 'I-STATE', 'I-GENDER',
            'I-COUNTY', 'I-JOBTYPE', 'I-JOBTITLE',
            'B-TIME', 'I-TIME', 'TIME',
            'B-MAC', 'I-MAC', 'MAC',
            'B-VEHICLEVRM', 'I-VEHICLEVRM', 'VEHICLEVRM',
            'B-DOB', 'I-DOB', 'DOB',
            'B-IP', 'I-IP', 'IP',
            'B-CURRENCYNAME', 'I-CURRENCYNAME', 'CURRENCYNAME',
            'B-CURRENCY', 'I-CURRENCY','CURRENCY',
            'B-BIC', 'I-BIC', 'BIC',
            'B-GENDER', 'I-GENDER', 'GENDER',
            'B-BITCOINADDRESS', 'I-BITCOINADDRESS', 'BITCOINADDRESS',
            'B-SECONDARYADDRESS', 'I-SECONDARYADDRESS', 'SECONDARYADDRESS',
            'B-IBAN', 'I-IBAN', 'IBAN',
            'B-IPV6', 'I-IPV6', 'IPV6',
            'B-COUNTY', 'I-COUNTY', 'COUNTY',
            'B-AMOUNT', 'I-AMOUNT', 'AMOUNT',
            'B-ETHEREUMADDRESS', 'I-ETHEREUMADDRESS', 'ETHEREUMADDRESS',
            'B-PASSWORD', 'I-PASSWORD', 'PASSWORD',
            'B-HEIGHT', 'I-HEIGHT', 'HEIGHT',
            'B-URL', 'I-URL', 'URL',
            'B-MASKEDNUMBER', 'I-MASKEDNUMBER', 'MASKEDNUMBER',
            'B-PREFIX', 'I-PREFIX', 'PREFIX',
            'B-BUILDINGNUMBER', 'I-BUILDINGNUMBER', 'BUILDINGNUMBER',
            'B-CURRENCYSYMBOL', 'I-CURRENCYSYMBOL', 'CURRENCYSYMBOL',
            'B-EYECOLOR', 'I-EYECOLOR', 'EYECOLOR',
            'B-VEHICLEVIN', 'I-VEHICLEVIN', 'VEHICLEVIN',
            'B-SSN', 'I-SSN', 'SSN',
            'B-STREET', 'I-STREET', 'STREET',
            'B-IPV4', 'I-IPV4', 'IPV4',
            'B-ORDINALDIRECTION', 'I-ORDINALDIRECTION', 'ORDINALDIRECTION',
            'B-USERAGENT', 'I-USERAGENT', 'USERAGENT',
            'B-LITECOINADDRESS', 'I-LITECOINADDRESS', 'LITECOINADDRESS',
            'B-CURRENCYCODE', 'I-CURRENCYCODE', 'CURRENCYCODE',
            'B-NEARBYGPSCOORDINATE', 'I-NEARBYGPSCOORDINATE', 'NEARBYGPSCOORDINATE',
            'B-DATE', 'I-DATE', 'DATE',
            'B-PIN', 'I-PIN', 'PIN'
        ]

        # Initialize a list to store final tokens
        final_tokens = []

        # Iterate over tokens and predicted labels
        for token, label in zip(tokens, predicted_labels):
            # Append label if it's not 'O' or in the list of ignore labels, otherwise append the token
            if label != 'O' and label not in ignore_labels:
                final_tokens.append(label)
            else:
                final_tokens.append(token)

        # Extract tokens that are not continuation tokens (not starting with 'I-')
        list_sentence = [word for word in final_tokens if not word.startswith('I-')]
        
        # Convert the list of tokens back into a string
        final_sentence = tokenizer.convert_tokens_to_string(list_sentence)

        return final_sentence


        # Initialization
    if 'session_id' not in st.session_state:
        # Generate a new session ID if not already present
        st.session_state.session_id = str(uuid.uuid4())
        # Initialize chat history
        st.session_state.chat_history = []

    if "vector_store" not in st.session_state:
        # Load document embeddings for both opinions (Demb) and law documents (Lemb) from local files
        Demb = FAISS.load_local("C:\\Users\\amine\\chatbot\\vectors\\Demb", HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5",model_kwargs={"device":"cuda"}), allow_dangerous_deserialization=True)
        Lemb = FAISS.load_local("C:\\Users\\amine\\chatbot\\vectors\\Lemb", HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device":"cuda"}), allow_dangerous_deserialization=True)
        st.session_state.vector_store = (Demb, Lemb)

    if "llm" not in st.session_state:
        # Initialize the language model (ChatOllama)
        st.session_state.llm = ChatOllama(model="mistral",
                                        temperature=0.2,
                                        num_ctx=32768,
                                        top_k=10,
                                        top_p=0.5,
                                        keep_alive=-1,
                                        num_predict=6000,
                                        num_batch=1024)

    if 'conversation_names' not in st.session_state:
        # Initialize conversation names dictionary
        st.session_state.conversation_names = {}

    if "PIImodel" not in st.session_state:
        # Load the fine-tuned model for token classification of personally identifiable information (PII)
        st.session_state.PIImodel = DistilBertForTokenClassification.from_pretrained('C:\\Users\\amine\\chatbot\\src\\my_model')

    if "PIItokenizer" not in st.session_state:
        # Load the tokenizer for the fine-tuned PII token classification model
        st.session_state.PIItokenizer = AutoTokenizer.from_pretrained("C:\\Users\\amine\\chatbot\\src\\my_tokenizer")

    # Display the title of the chatbot
    st.title("Hello, I am a juridical bot. How can I help you?")

    # Sidebar
    with st.sidebar:
        # Apply custom CSS styles
        st.markdown("""
                    
        <style>
        section[data-testid="stSidebar"] div.stButton button {
        background-color: grey;
        width: 250px;
        }
        section[data-testid="stSidebar"] div.stMultiSelect div[data-baseweb="select"] {
        width: 250px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header for the sidebar
        st.header("Thank you for using our chatbot")
        
        # Button to log out
        if st.button('Log Out'):
            # Log out the user and return to the main page
            st.session_state.logged_in = False
            st.session_state.page = 'main'
            st.experimental_rerun()
        
        # Divider
        st.markdown("---")
        
        # Header for conversations
        st.header("Conversations")
        
        # Button to start a new conversation
        if st.button("New Conversation"):
            # Generate a new session ID and reset chat history
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []

        # Connect to the database to fetch conversation IDs and their latest timestamps
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT session_id, MAX(created_at) AS latest_timestamp
            FROM messages
            WHERE user_id = %s
            GROUP BY session_id
            ORDER BY latest_timestamp DESC
        """, (user_id,))
        session_ids_with_timestamps = cur.fetchall()
        cur.close()
        conn.close()

        # Initialize a list to store conversation names
        conversation_names = []

        # Iterate over fetched session IDs and their timestamps
        for session_id, timestamp in session_ids_with_timestamps:
            if session_id in st.session_state.conversation_names:
                # Retrieve conversation name if already generated
                conversation_name = st.session_state.conversation_names[session_id]
            else:
                # Load conversation history and generate conversation name using the language model (llm)
                conversation_history = load_conversation(session_id)
                conversation_name = generate_conversation_name_with_llm(session_id, conversation_history, st.session_state.llm)

            # Convert timestamp to a readable format
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

            # Add the conversation name with timestamp to the list
            conversation_names.append((session_id, f"{conversation_name} ({timestamp_str})"))

        # Display the conversations in chronological order
        for session_id, conversation_name in conversation_names:
            if st.button(conversation_name):
                # Set the selected conversation as the current session and load its history
                st.session_state.session_id = session_id
                conversation_history = load_conversation(session_id)
                st.session_state.chat_history = conversation_history

        # Multi-select dropdown to delete selected conversations
        selected_conversation_ids = st.multiselect("Select conversations to delete", [conversation_name for _, conversation_name in conversation_names])

        # Button to delete selected conversations
        if st.button("Delete Selected"):
            # Connect to the database to delete selected conversations
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            cur = conn.cursor()
            for conversation_id in selected_conversation_ids:
                session_id = next((session_id for session_id, conversation_name in conversation_names if conversation_name == conversation_id), None)
                if session_id is not None:
                    # Delete messages associated with the selected conversation
                    cur.execute("""
                        DELETE FROM messages
                        WHERE session_id = %s
                    """, (session_id,))
                    conn.commit()

                    if session_id == st.session_state.session_id:
                        # Reset chat history if the deleted conversation was the current session
                        st.session_state.chat_history = []

                    if session_id in st.session_state.conversation_names:
                        # Remove conversation name from the dictionary if present
                        del st.session_state.conversation_names[session_id]

            cur.close()
            conn.close()
            st.experimental_rerun()

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            # Display AI's message
            with st.chat_message("AI"):
                st.markdown(re.sub(r'(?<!\\)\$', '\\$', message.content))
        elif isinstance(message, HumanMessage):
            # Display user's message
            with st.chat_message("Human"):
                st.markdown(re.sub(r'(?<!\\)\$', '\\$', message.content))

    # Input field for user query
    query = st.chat_input("Enter your question here:")
    if query:
        # Tokenize user's query to handle personally identifiable information (PII)
        new_query = Tokenize_PII_data(st.session_state.PIImodel, st.session_state.PIItokenizer, query)
        # Append the tokenized query to the chat history as a HumanMessage
        st.session_state.chat_history.append(HumanMessage(content=new_query))
        # Display user's message instantly
        with st.chat_message("Human"):
            st.markdown(re.sub(r'(?<!\\)\$', '\\$', new_query))

        # Generate response based on user's query
        with st.spinner("Generating response..."):
            Demb, Lemb = st.session_state.vector_store
            # Retrieve relevant opinions and law documents
            op = retrieve_opinion(query, Demb)
            Law = retrieve_Law(query, Lemb)
            # Create a response using RAG (Retrieve, Attend, Generate) chain
            rag_chain = create_rag_chain(op, Law, st.session_state.chat_history)
            result = rag_chain.invoke(query)
        # Append AI's response to the chat history
        st.session_state.chat_history.append(AIMessage(content=result['text']))
        print(result['text'])
        # Display AI's response after generation
        with st.chat_message("AI"):
            st.markdown(re.sub(r'(?<!\\)\$', '\\$', result['text']))

        # Save the user's query and AI's response to the database
        save_message(st.session_state.session_id, 'user', new_query)
        save_message(st.session_state.session_id, 'ai', result['text'])


