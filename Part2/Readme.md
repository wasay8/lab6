# Data Collection/ Processing/ Storage

Required package: 

langchain 0.0.319 
PyPDF 3.0.1 
streamlit 1.27.2 
sentence-transformers 2.2.2
faiss-cpu 1.7.4

Make sure that your computer install llama before run this project.
## Workflow

To run the chatbox, please enter the code below:

> streamlit run chatbox.py

Then you will see the Local URL

![image](https://github.com/thoughtfuldata/DSCI560-project/assets/55038803/74d1fadd-ad3e-4dda-b023-4eab361c7ef3)


Open the URL in your browser. Now you can upload the reference PDF and ask question to AI.

![image](https://github.com/thoughtfuldata/DSCI560-project/assets/55038803/0e8eb58b-d155-4562-a994-a89cfe9fd5e5)


Left panel is a section for user upload reference PDF. And right panel is a section that chat with AI.

It include four step: PDF extraction, Text chunks, Vectorizing, Conversation Chain

The terminal will also show some tips:

![image](https://github.com/thoughtfuldata/DSCI560-project/assets/55038803/ad8b3b90-66fe-4ea6-9493-53b2a6d565b5)

In this part, for chating model. We use llama2 to instead OpenAI. And for embedding model, we choose BAAI. Now we don't need OPENAI_API_KEY because we change all model to open-source model.




















